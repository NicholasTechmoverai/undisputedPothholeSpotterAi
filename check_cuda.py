import torch
print("cuda available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
import torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__)