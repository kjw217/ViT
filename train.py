# 실행: torchrun --nproc_per_node=4 train.py --model_name vit-s --batch 128
# 특정 GPU만 쓰려면: CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from datasets import load_dataset
from data import Imagenet1K
from tqdm import tqdm
import csv
import math
import os
import time
import argparse
from model_variants import vit_small, vit_base, vit_large
import warnings
warnings.filterwarnings("ignore", message="Corrupt EXIF data")

parser = argparse.ArgumentParser(description="ViT ImagenNet")
parser.add_argument('--model_name', type=str, default='vit-s')
parser.add_argument('--epoch', type=int, default=90)
parser.add_argument('--batch', type=int, default=128, help="GPU 1장당 배치. 실제 배치 = batch * GPU 수")
parser.add_argument('-lr', type=float, default=3e-3)
parser.add_argument('--optimizer', type=str, default="AdamW")
args = parser.parse_args()

class CosineWithWarmup(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, training_steps, num_cycles, last_epoch=-1):
        '''
        optimizer : target optimizer to apply learning rate scheduler
        warmup_steps : step size of warming up the learning rate
        training_steps : total training step
        last_epoch : last epoch of training
        '''
        def lr_lambda(cur_step):
            if cur_step < warmup_steps:
                return float(cur_step) / float(max(1.0, warmup_steps))
            progress = float(cur_step - warmup_steps) / float(max(1, training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)

def train() :
    # [DDP 1] 프로세스 그룹 초기화
    # torchrun이 GPU 수만큼 이 스크립트를 따로 실행하고, 각 프로세스에 RANK/LOCAL_RANK/WORLD_SIZE 환경변수를 넣어준다.
    # 즉 아래 코드는 전부 "GPU 1장을 맡은 프로세스 하나"의 시점에서 읽으면 된다.
    dist.init_process_group(backend="nccl")     # nccl = NVIDIA GPU 간 통신 라이브러리
    local_rank = int(os.environ["LOCAL_RANK"])  # 이 서버 안에서의 내 번호 = 내가 쓸 GPU 번호
    is_main = dist.get_rank() == 0              # 출력/저장은 0번 프로세스만 한다 (안 그러면 GPU 수만큼 중복됨)
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    # [DDP 2] 모델을 내 GPU에 올리고 DDP로 감싼다
    # 생성 시 0번의 가중치를 전체에 복사(broadcast)하므로 프로세스마다 초기값이 달라도 시작점은 같아진다.
    # 이후 backward() 때 각 GPU의 gradient를 자동으로 평균(all-reduce)해서, 모든 GPU가 같은 가중치를 유지한다.
    model = {"vit-s": vit_small, "vit-b": vit_base, "vit-l": vit_large}[args.model_name]
    model = DDP(model.to(device), device_ids=[local_rank])

    ds = load_dataset("ILSVRC/imagenet-1k", num_proc=8)
    train_dataset = Imagenet1K(ds, datatype="train", image_size=224)
    val_dataset = Imagenet1K(ds, datatype="validation", image_size=224)

    # [DDP 3] DistributedSampler: 데이터셋을 GPU 수로 쪼개 각 프로세스가 서로 다른 부분만 보게 한다
    # 셔플은 sampler가 담당하므로 DataLoader에는 shuffle=True를 주면 안 된다.
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, sampler=train_sampler, pin_memory=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, sampler=val_sampler, pin_memory=True, num_workers=8)

    if args.optimizer == "AdamW" :
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=0.3)
    elif args.optimizer == "SGD" :
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=0.3)
    # 데이터가 쪼개졌으므로 len(train_dataloader)는 이미 "GPU 1장 기준" 스텝 수다
    scheduler = CosineWithWarmup(optimizer=optimizer, warmup_steps=10000, training_steps=args.epoch*len(train_dataloader), num_cycles=1)
    gradscaler = GradScaler("cuda")

    # 학습 로그: 0번 프로세스만 기록한다. 실행마다 새 파일을 만들어서 이전 로그를 덮어쓰지 않는다
    # iter 행은 1000 iter 평균 train loss, val_loss가 채워진 행은 그 epoch의 평균 train loss다
    if is_main :
        os.makedirs("result", exist_ok=True)
        log_file = open(f"result/log_{args.model_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv", "w", newline="")
        logger = csv.writer(log_file)
        logger.writerow(["epoch", "iter", "train_loss", "val_loss", "lr", "elapsed_sec"])
        start_time = time.time()

    def write_log(epoch, iter, train_loss, val_loss="") :
        logger.writerow([epoch, iter, train_loss, val_loss, scheduler.get_last_lr()[0], round(time.time() - start_time)])
        log_file.flush()  # 학습이 중간에 죽어도 여기까지의 로그는 남도록 바로 파일에 쓴다

    mixup = torch.distributions.Beta(0.2, 0.2)  # 0.2면 섞는 비율이 대부분 0이나 1 근처로 나온다

    iter_loss = 0
    min_val_loss = 1e5
    for epoch in range(args.epoch) :
        # [DDP 4] 매 epoch 호출해야 셔플 순서가 바뀐다 (빼먹으면 매 epoch 같은 순서로 학습)
        train_sampler.set_epoch(epoch)

        epoch_loss = 0
        model.train()
        for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not is_main) :
            image, label = data
            image, label = image.to(device), label.to(device)

            # Mixup: 배치를 뒤집어 짝지은 두 이미지를 섞고, 라벨도 같은 비율로 섞는다 (학습 때만 적용)
            lam = mixup.sample().item()
            image = lam * image + (1 - lam) * image.flip(0)
            label = lam * label + (1 - lam) * label.flip(0)

            optimizer.zero_grad()

            with autocast(device_type="cuda") :
                pred = model(image)
                loss = F.cross_entropy(pred, label, label_smoothing=0.1)
            gradscaler.scale(loss).backward()  # 여기서 GPU 간 gradient 평균이 자동으로 일어난다
            gradscaler.unscale_(optimizer)     # clip은 스케일을 되돌린 gradient에 해야 한다
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            gradscaler.step(optimizer)
            gradscaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            iter_loss += loss.item()
            if i % 1000 == 999 :
                if is_main :
                    print(f"{epoch} epoch / {i+1} iter : {iter_loss / (1000)}")
                    write_log(epoch, i+1, iter_loss / 1000)
                iter_loss = 0
        # train loss는 0번 GPU가 본 데이터 기준 값이다 (로그용이라 굳이 합치지 않음)
        train_loss = epoch_loss / (i + 1)
        if is_main : print(f"{epoch} epoch loss : {train_loss}")
        model.eval()
        with torch.no_grad() :
            valid_loss = 0
            for i, val_data in tqdm(enumerate(val_dataloader), total=len(val_dataloader), disable=not is_main) :
                image, label = val_data
                image, label = image.to(device), label.to(device)

                with autocast(device_type="cuda") :
                    pred = model(image)
                    loss = F.cross_entropy(pred, label)
                valid_loss += loss.item()

        # [DDP 5] 각 GPU는 검증셋의 일부만 봤으므로 전체 평균을 구한다
        # all_reduce 후에는 모든 프로세스가 같은 값을 가진다
        val_loss = torch.tensor(valid_loss/(i+1), device=device)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = val_loss.item()

        if is_main :
            print(f"{epoch} validation loss : {val_loss}")
            write_log(epoch, len(train_dataloader), train_loss, val_loss)
            if val_loss < min_val_loss :
                min_val_loss = val_loss
                # [DDP 6] 가중치는 모든 GPU가 같으니 0번만 저장한다
                # 실제 모델은 model.module에 있다 (model로 저장하면 key에 "module."이 붙어서 DDP 없이 로드할 때 꼬인다)
                os.makedirs("result/model_chkpoint", exist_ok=True)
                torch.save(model.module.state_dict(), f"result/model_chkpoint/epoch{epoch}_{str(val_loss).replace(".", "_")}.pth")

    # [DDP 7] 프로세스 그룹 정리
    dist.destroy_process_group()

if __name__ == "__main__" :
    train()
