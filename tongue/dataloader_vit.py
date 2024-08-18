import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class TongueDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.features = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['image_path'])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # labels = torch.tensor(self.data.iloc[idx][self.features], dtype=torch.float32)
        rows = self.data.iloc[idx].loc[self.features]
        labels = torch.tensor(rows.tolist(), dtype=torch.float32)
        
        return image, labels

def get_dataloaders(train_csv, val_csv, img_dir, batch_size=32, num_workers=4):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
        # transforms.RandomApply([transforms.RandomRotation(5)], p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.05, contrast=0, saturation=0, hue=0)
        ], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.5))], p=0.5),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # bicubic
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if train_csv is not None:
        train_dataset = TongueDataset(train_csv, img_dir, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        train_loader = None

    val_dataset = TongueDataset(val_csv, img_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader