import os
import shutil
from google.colab import drive, files

def setup_directories():
    # Mount Google Drive nếu chưa mount
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    
    # Tạo thư mục train và test
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    
    # Tạo các thư mục con trong train
    train_subfolders = [
        "nấm mỡ",
        "bào ngư xám + trắng",
        "Đùi gà Baby (cắt ngắn)",
        "linh chi trắng"
    ]
    
    for folder in train_subfolders:
        os.makedirs(os.path.join('train', folder), exist_ok=True)
    
    print("\nĐã tạo cấu trúc thư mục:")
    print("train/")
    for folder in train_subfolders:
        print(f"  ├── {folder}/")
    print("test/")

def check_data():
    # Kiểm tra thư mục train
    train_path = './train'
    if os.path.exists(train_path):
        print("\nKiểm tra thư mục train:")
        total_images = 0
        for folder in os.listdir(train_path):
            folder_path = os.path.join(train_path, folder)
            if os.path.isdir(folder_path):
                num_files = len([f for f in os.listdir(folder_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total_images += num_files
                print(f"- {folder}: {num_files} ảnh")
        print(f"\nTổng số ảnh trong train: {total_images}")
    
    # Kiểm tra thư mục test
    test_path = './test'
    if os.path.exists(test_path):
        num_test = len([f for f in os.listdir(test_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Số ảnh trong test: {num_test}")

def main():
    print("Thiết lập thư mục dữ liệu...")
    setup_directories()
    
    print("\nBạn cần upload dữ liệu vào các thư mục tương ứng:")
    print("1. Upload ảnh train vào thư mục 'train/tên_loại_nấm/'")
    print("2. Upload ảnh test vào thư mục 'test/'")
    print("\nSau khi upload xong, chạy lệnh sau để kiểm tra:")
    print("check_data()")

if __name__ == '__main__':
    main() 