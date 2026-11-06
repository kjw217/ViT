import os
import PIL
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Imagenet1K(Dataset) :
    def __init__(self, ds, datatype:str, image_size:int):
        super().__init__()
        self.ds = ds[datatype]
        if datatype == "train" :
            # 불러올 때마다 랜덤 변형 -> 같은 이미지도 epoch마다 다르게 보인다
            resize = [transforms.RandomResizedCrop(image_size),
                      transforms.RandomHorizontalFlip(),
                      # 논문의 randaug(2, 10): 변형 2개를 강도 10으로 적용 (강도 단계를 0~10으로 맞췄다)
                      transforms.RandAugment(num_ops=2, magnitude=10, num_magnitude_bins=11)]
        else :
            # 평가 기준이 흔들리면 안 되므로 고정 변환만 쓴다 (비율 유지 resize 후 중앙 crop)
            resize = [transforms.Resize(int(image_size * 256 / 224)),
                      transforms.CenterCrop(image_size)]
        self.pil2tensor = transforms.Compose(resize + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.ds.__len__()

    def __getitem__(self, index):
        row = self.ds[index]  # 한 행을 한 번에 읽는다 (컬럼별로 따로 읽으면 NAS 접근이 늘어나 느리다)
        image = self.pil2tensor(row["image"].convert("RGB"))
        label = F.one_hot(torch.tensor(row["label"]), 1000).to(dtype=torch.float16)
        return image, label
    