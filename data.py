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
        self.pil2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return self.ds.__len__()

    def __getitem__(self, index):
        image = self.pil2tensor(self.ds["image"][index].convert("RGB"))
        label = F.one_hot(torch.tensor(self.ds["label"][index]), 1000).to(dtype=torch.float16)
        return image, label
    