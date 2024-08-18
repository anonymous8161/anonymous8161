import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class TongueDataset(Dataset):
    def __init__(self, csv_file, whole_img_dir, body_img_dir, edge_img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.whole_img_dir = whole_img_dir
        self.body_img_dir = body_img_dir
        self.edge_img_dir = edge_img_dir
        self.transform = transform
        self.features = ['TonguePale', 'TipSideRed', 'RedSpot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load whole image
        whole_img_name = os.path.join(self.whole_img_dir, self.data.iloc[idx]['image_path'])
        whole_image = Image.open(whole_img_name).convert('RGB')
        
        # Load body image
        body_img_name = os.path.join(self.body_img_dir, self.data.iloc[idx]['body_image_path'])
        body_image = Image.open(body_img_name).convert('RGB')
        
        # Load edge image
        edge_img_name = os.path.join(self.edge_img_dir, self.data.iloc[idx]['edge_image_path'])
        edge_image = Image.open(edge_img_name).convert('RGB')
        
        if self.transform:
            whole_image = self.transform(whole_image)
            body_image = self.transform(body_image)
            edge_image = self.transform(edge_image)
        
        rows = self.data.iloc[idx].loc[self.features]
        labels = torch.tensor(rows.tolist(), dtype=torch.float32)
        
        color_label = torch.zeros(4, dtype=torch.float32)
        if labels[0] == 1 or labels[6] == 1:  # TonguePale or FurThick
            color_label[0] = 1  # White
        elif labels[7] == 1:  # FurYellow
            color_label[1] = 1  # Yellow
        elif labels[3] == 1:  # Ecchymosis
            color_label[2] = 1  # Black
        elif labels[1] == 1 or labels[2] == 1:  # TipSideRed or RedSpot
            color_label[3] = 1  # Red
        
        fur_label = torch.tensor([1.0 if labels[2] == 1 or labels[7] == 1 or labels[6] == 1 else 0.0])
        
        return whole_image, body_image, edge_image, labels, color_label, fur_label

def get_dataloaders(train_csv, val_csv, whole_img_dir, body_img_dir, edge_img_dir, batch_size=32, num_workers=4):
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
        train_dataset = TongueDataset(train_csv, whole_img_dir, body_img_dir, edge_img_dir, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        train_loader = None

    val_dataset = TongueDataset(val_csv, whole_img_dir, body_img_dir, edge_img_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader