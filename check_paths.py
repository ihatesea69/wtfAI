import os

def check_paths():
    # Kiểm tra thư mục hiện tại
    current_dir = os.getcwd()
    print(f"Thư mục hiện tại: {current_dir}")
    
    # Các đường dẫn cần kiểm tra
    paths_to_check = {
        'Code': '/content/wtfAI',
        'Training Data': '/content/drive/MyDrive/Training Data',
        'Train': '/content/drive/MyDrive/Training Data/train',
        'Test': '/content/drive/MyDrive/Training Data/test'
    }
    
    print("\nKiểm tra các đường dẫn:")
    for name, path in paths_to_check.items():
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
            if name == 'Train' and os.path.isdir(path):
                print("  Các thư mục con:")
                for folder in os.listdir(path):
                    folder_path = os.path.join(path, folder)
                    if os.path.isdir(folder_path):
                        num_files = len([f for f in os.listdir(folder_path) 
                                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        print(f"  - {folder}: {num_files} ảnh")
        else:
            print(f"✗ {name}: Không tìm thấy {path}")

if __name__ == '__main__':
    check_paths() 