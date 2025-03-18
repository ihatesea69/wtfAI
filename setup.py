from google.colab import drive
import os

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Di chuyển đến thư mục project
os.chdir('/content/drive/MyDrive/wtfAI')
print(f"Current directory: {os.getcwd()}")

# In ra cấu trúc thư mục
print("\nCấu trúc thư mục:")
print("Training Data/train:")
if os.path.exists("Training Data/train"):
    for folder in os.listdir("Training Data/train"):
        if os.path.isdir(os.path.join("Training Data/train", folder)):
            num_files = len([f for f in os.listdir(os.path.join("Training Data/train", folder)) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"- {folder}: {num_files} ảnh") 