from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class MangaDataset(Dataset):
    def __init__(self, root_color, root_blackwhite, transform=None):
        self.root_color = root_color
        self.root_blackwhite = root_blackwhite
        self.transform = transform

        self.color_images = os.listdir(root_color)
        self.blackwhite_images = os.listdir(root_blackwhite)
        self.length_dataset = max(len(self.color_images), len(self.blackwhite_images)) 
        self.color_len = len(self.color_images)
        self.blackwhite_len = len(self.blackwhite_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        color_img = self.color_images[index % self.color_len]
        blackwhite_img = self.blackwhite_images[index % self.blackwhite_len]

        color_path = os.path.join(self.root_color, color_img)
        blackwhite_path = os.path.join(self.root_blackwhite, blackwhite_img)

        color_img = np.array(Image.open(color_path).convert("RGB"))
        blackwhite_img = np.array(Image.open(blackwhite_path).convert("L"))

        if self.transform:
            augmentations = self.transform(image=color_img, image0=blackwhite_img)
            color_img = augmentations["image"]
            blackwhite_img = augmentations["image0"]

        return color_img, blackwhite_img





