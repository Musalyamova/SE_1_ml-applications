import os
os.environ["HF_HOME"] = "D:\\hf-cache"

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

model = torch.hub.load('pytorch/vision', 'fcn_resnet50', pretrained=True)
model.eval()

image_path = r"D:\ml-applications\image\test1.png"
input_image = Image.open(image_path).convert("RGB")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(input_batch)['out'][0]

output_predictions = output.argmax(0).byte().cpu().numpy()

classes = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]

unique, counts = np.unique(output_predictions, return_counts=True)
print("Объекты на фото:")
for cls_id, count in zip(unique, counts):
    print(f"{classes[cls_id]}: {count} пикселей ({count / output_predictions.size * 100:.1f}%)")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title("Оригинал")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output_predictions, cmap="tab20")
plt.title("Сегментация FCN")
plt.axis("off")

plt.tight_layout()
plt.show()