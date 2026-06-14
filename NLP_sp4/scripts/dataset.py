import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoTokenizer
from scripts.config import CFG

# Трансформации
train_transform = A.Compose([
    A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.Normalize(),
    ToTensorV2()
])

valid_transform = A.Compose([
    A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# Токенизатор (один на весь проект)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class DishDataset(Dataset):
    def __init__(self, dataframe, transform=None, max_len=128):
        self.df = dataframe
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(CFG.IMAGES_PATH, str(row["dish_id"]), "rgb.png")
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]

        text = row["ingredients_text"]
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "mass": torch.tensor(row["total_mass"], dtype=torch.float),
            "target": torch.tensor(row["total_calories"], dtype=torch.float)
        }
