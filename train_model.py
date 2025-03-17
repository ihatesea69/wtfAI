import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Định nghĩa model lớn hơn
class MushroomCNN(nn.Module):
    def __init__(self):
        super(MushroomCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Dataset class
class MushroomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Label mapping
        self.label_dict = {
            "nấm mỡ": 0,
            "Nấm bào ngư": 1,
            "Nấm đùi gà": 2,
            "nấm linh chi trắng": 3
        }
        
        # Load data
        for label_folder in os.listdir(folder_path):
            if label_folder in self.label_dict:
                folder = os.path.join(folder_path, label_folder)
                for img_name in os.listdir(folder):
                    self.images.append(os.path.join(folder, img_name))
                    self.labels.append(self.label_dict[label_folder])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_acc = 0.0
    scaler = GradScaler()  # For mixed precision training
    
    # In thông tin GPU trước khi bắt đầu train
    if torch.cuda.is_available():
        print(f"\nGPU Memory trước khi train:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Sử dụng tqdm với thông tin chi tiết hơn
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Mixed precision training
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass với mixed precision
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Tính toán metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Cập nhật progress bar
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'GPU Mem': f'{torch.cuda.memory_allocated()/1024**2:.1f}MB' if torch.cuda.is_available() else 'N/A',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        # In kết quả chi tiết
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if torch.cuda.is_available():
            print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')
        
        print('-' * 60)
        
        # Xóa cache GPU sau mỗi epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    # Kiểm tra GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if torch.cuda.is_available():
        # Tối ưu hóa CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Cho phép TF32 trên Ampere GPUs
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU được sử dụng: {torch.cuda.get_device_name(0)}")
        print(f"Số lượng GPU: {torch.cuda.device_count()}")
    
    # Định nghĩa transforms
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Tăng kích thước ảnh
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = MushroomDataset('./train', transform=train_transform)
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Tăng batch size và số worker
    BATCH_SIZE = 128  # Tăng batch size lên tối đa
    NUM_WORKERS = 8   # Tăng số worker
    
    # Create data loaders với pin_memory=True để tăng tốc chuyển data lên GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model và chuyển lên GPU
    model = MushroomCNN().to(device)
    
    # Sử dụng One Cycle Policy với learning rate cao hơn
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=100,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    criterion = nn.CrossEntropyLoss()
    
    print("\nBắt đầu training với các thông số:")
    print(f"Image size: 64x64")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Số worker: {NUM_WORKERS}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Device: {device}")
    print("Mixed Precision Training: Enabled")
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, device=device)

if __name__ == '__main__':
    main() 