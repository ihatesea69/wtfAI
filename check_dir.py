import os

def print_directory_structure(startpath, level=0):
    for element in os.listdir(startpath):
        path = os.path.join(startpath, element)
        indent = '│   ' * level + '├── '
        if os.path.isdir(path):
            print(f"{indent}{element}/")
            print_directory_structure(path, level + 1)
        else:
            print(f"{indent}{element}")

# Kiểm tra thư mục hiện tại
current_dir = os.getcwd()
print(f"\nThư mục hiện tại: {current_dir}")

# Liệt kê các file và thư mục
print("\nCác file và thư mục trực tiếp:")
for item in os.listdir('.'):
    if os.path.isdir(item):
        print(f"[DIR] {item}")
    else:
        print(f"[FILE] {item}")

# In cấu trúc thư mục chi tiết
print("\nCấu trúc thư mục chi tiết:")
print_directory_structure('.') 