import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    
# Thử nghiệm với một tensor
x = torch.rand(5, 3)
print("\nTensor trên CPU:")
print(x)

if torch.cuda.is_available():
    x = x.cuda()
    print("\nTensor trên GPU:")
    print(x) 