import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Định nghĩa model giống như trong train_model.py
class MushroomCNN(nn.Module):
    def __init__(self):
        super(MushroomCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def preprocess_image(image_path, transform):
    """
    Tiền xử lý ảnh cho inference
    """
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)  # Thêm batch dimension
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
        return None

def main():
    print("Bắt đầu quá trình tạo file submission...")
    
    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Mapping từ chỉ số dự đoán của model sang mã loại theo đề bài
    label_mapping = {0: 1, 1: 2, 2: 3, 3: 0}
    print("Mapping dự đoán:", label_mapping)
    
    # Transform cho inference
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load model
    model = MushroomCNN().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    # Đường dẫn chứa ảnh test - cập nhật đường dẫn tuyệt đối
    test_folder = '/content/drive/MyDrive/Training Data/test'
    if not os.path.exists(test_folder):
        raise FileNotFoundError(f"Không tìm thấy thư mục test tại {test_folder}")
    
    # Dự đoán trên tập test
    predictions = []
    image_ids = []
    
    print("Đang thực hiện dự đoán...")
    with torch.no_grad():
        for i in tqdm(range(1, 201)):
            image_name = f"{i:03d}.jpg"
            image_path = os.path.join(test_folder, image_name)
            
            if os.path.exists(image_path):
                # Tiền xử lý ảnh
                image = preprocess_image(image_path, test_transform)
                if image is not None:
                    # Dự đoán
                    image = image.to(device)
                    output = model(image)
                    pred = output.argmax(dim=1).item()
                    predictions.append(label_mapping[pred])
                    image_ids.append(str(i))
            else:
                print(f"Warning: Không tìm thấy ảnh {image_name}")
    
    # Kiểm tra kết quả dự đoán
    unique_labels, counts = torch.unique(torch.tensor(predictions), return_counts=True)
    print("\nThống kê kết quả dự đoán:")
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        print(f"Loại {label}: {count} ảnh")
    
    # Tạo DataFrame cho submission
    df_submission = pd.DataFrame({
        'id': image_ids,
        'type': predictions
    })
    
    # Kiểm tra định dạng dữ liệu
    print("\nKiểm tra định dạng dữ liệu:")
    print("5 dòng đầu của file submission:")
    print(df_submission.head())
    
    # Lưu file submission
    submission_file = 'submission.csv'
    df_submission.to_csv(submission_file, index=False)
    print(f"\nFile submission đã được lưu với tên: {submission_file}")
    print(f"Tổng số dòng trong file: {len(df_submission)}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Lỗi: {str(e)}")
