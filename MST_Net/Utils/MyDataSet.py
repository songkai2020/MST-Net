from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MyDataSetimg2img(Dataset):
    def __init__(self,img_path,label_path,transform = None):
        super().__init__()
        self.img_path = img_path
        self.label_path = label_path
        self.trasform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img = self.trasform(Image.open(self.img_path[item]).convert('L'))
        label = self.trasform(Image.open(self.label_path[item]).convert('L'))
        return img,label

