import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from datasets import load_dataset
from huggingface_hub import login
from data import Imagenet1K
from tqdm import tqdm
import math
import os

login(token=os.environ.get("HF_TOKEN"))

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

def train(model:nn.Module, epochs, batch_size, base_lr, optimizer_name, device="cpu") :
    model = model.to(device)

    ds = load_dataset("ILSVRC/imagenet-1k")

    train_dataset = Imagenet1K(ds, datatype="train", image_size=224)
    val_dataset = Imagenet1K(ds, datatype="validation", image_size=224)
    test_dataset = Imagenet1K(ds, datatype="test", image_size=224)
    if device == "cuda" :
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)
    else :
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    if optimizer_name == "AdamW" :
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr, weight_decay=0.3)
    elif optimizer_name == "SGD" :
        optimizer = torch.optim.SGD(params=model.parameters(), lr=base_lr, weight_decay=0.3)
    scheduler = CosineWithWarmup(optimizer=optimizer, warmup_steps=10000, training_steps=epochs*(len(train_dataset)//batch_size), num_cycles=1)
    gradscaler = GradScaler(device)

    record = []

    iter_loss = 0
    min_val_loss = 1e5
    for epoch in range(epochs) :
        epoch_loss = 0
        loss_record = []
        model.train()
        for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)) :
            image, label = data
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device) :
                pred = model(image)
                loss = F.cross_entropy(pred, label, label_smoothing=0.1)
            gradscaler.scale(loss).backward()

            gradscaler.step(optimizer)
            gradscaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            iter_loss += loss.item()
            if i % 1000 == 999 :
                print(f"{epoch} epoch / {i+1} iter : {iter_loss / (1000)}")
                iter_loss = 0
        print(f"{epoch} epoch loss : {epoch_loss / (i + 1)}")
        loss_record.append(epoch_loss/(i+1))
        model.eval()
        with torch.no_grad() :
            valid_loss = 0
            for i, val_data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)) :
                image, label = val_data
                image, label = image.to(device), label.to(device)

                pred = model(image)
                loss = F.cross_entropy(pred, label)
                valid_loss += loss.item()
            print(f"{epoch} validation loss : {valid_loss/(i+1)}")
            loss_record.append(valid_loss/(i+1))
            if valid_loss/(i+1) < min_val_loss :
                torch.save(model.state_dict(), f"/result/model_chkpoint/epoch{epoch}_{str(valid_loss/(i+1)).replace(".", "_")}.pth")
        record.append(loss_record)
        loss_record = []

if __name__ == "__main__" :
    from model_variants import vit_small
    
    train(model=vit_small, epochs=1, batch_size=12, base_lr=1e-3, optimizer_name="AdamW")
