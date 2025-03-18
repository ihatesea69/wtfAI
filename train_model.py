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
        
        # Tự động tạo label_dict từ tên thư mục
        folders = sorted([f for f in os.listdir(folder_path) 
                        if os.path.isdir(os.path.join(folder_path, f))])
        self.label_dict = {folder: idx for idx, folder in enumerate(folders)}
        
        print("\nLabel mapping tự động:")
        for folder, idx in self.label_dict.items():
            print(f"- {folder}: {idx}")
        
        # Load data và in thông tin debug
        print("\nĐang load ảnh từ các thư mục:")
        for label_folder, label_idx in self.label_dict.items():
            folder_path_full = os.path.join(folder_path, label_folder)
            img_files = [f for f in os.listdir(folder_path_full) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"- {label_folder}: Tìm thấy {len(img_files)} ảnh")
            
            for img_name in img_files:
                self.images.append(os.path.join(folder_path_full, img_name))
                self.labels.append(label_idx)
        
        print(f"\nTổng số ảnh đã load: {len(self.images)}")
        print("Phân bố các lớp:")
        for label_name, label_id in self.label_dict.items():
            count = self.labels.count(label_id)
            print(f"- {label_name}: {count} ảnh")
        
        if len(self.images) == 0:
            raise ValueError("Không tìm thấy ảnh nào! Vui lòng kiểm tra lại thư mục.")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Lỗi khi load ảnh {img_path}: {str(e)}")
            # Trả về một ảnh ngẫu nhiên khác trong trường hợp lỗi
            return self.__getitem__((idx + 1) % len(self))

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_acc = 0.0
    
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
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Tính toán metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Cập nhật progress bar
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%',
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
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')
        
        print('-' * 60)

def main():
    # Kiểm tra GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Đường dẫn data - sử dụng đường dẫn tuyệt đối
    TRAIN_PATH = '/content/drive/MyDrive/Training Data/train'
    
    # Kiểm tra thư mục train
    print(f"\nKiểm tra thư mục {TRAIN_PATH}:")
    if os.path.exists(TRAIN_PATH):
        print(f"Tìm thấy thư mục: {TRAIN_PATH}")
        print("Các thư mục con:")
        total_images = 0
        for folder in os.listdir(TRAIN_PATH):
            folder_path = os.path.join(TRAIN_PATH, folder)
            if os.path.isdir(folder_path):
                images = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                num_files = len(images)
                total_images += num_files
                print(f"- {folder}: {num_files} ảnh")
        print(f"\nTổng số ảnh: {total_images}")
    else:
        raise FileNotFoundError(f"Không tìm thấy thư mục {TRAIN_PATH}")
    
    if total_images == 0:
        raise ValueError("Không tìm thấy ảnh nào trong thư mục train!")
    
    # Định nghĩa transforms - giảm kích thước ảnh xuống
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Giảm kích thước xuống
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Giảm kích thước xuống
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\nĐang load dataset...")
    train_dataset = MushroomDataset(TRAIN_PATH, transform=train_transform)
    print(f"Tổng số ảnh: {len(train_dataset)}")
    
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    print(f"Số ảnh train: {train_size}")
    print(f"Số ảnh validation: {val_size}")
    
    # Điều chỉnh batch size và workers
    BATCH_SIZE = 64  # Giảm batch size xuống
    NUM_WORKERS = 4  # Tăng số worker
    
    print(f"\nBatch size: {BATCH_SIZE}")
    print(f"Số worker: {NUM_WORKERS}")
    
    # Create data loaders
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
    
    # Create model và chuyển lên device
    model = MushroomCNN().to(device)
    
    # Sử dụng Adam optimizer với learning rate thấp hơn
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    
    print("\nBắt đầu training với các thông số:")
    print(f"Image size: 32x32")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Số worker: {NUM_WORKERS}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Device: {device}")
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device=device)

if __name__ == '__main__':
    main() 