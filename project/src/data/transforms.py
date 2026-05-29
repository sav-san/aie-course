
from torchvision import transforms

# Гиперпараметры из ноутбука (ячейка конфигурации)
IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]   # ImageNet
STD      = [0.229, 0.224, 0.225]   # ImageNet

# Точная копия val_tf из ноутбука
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
