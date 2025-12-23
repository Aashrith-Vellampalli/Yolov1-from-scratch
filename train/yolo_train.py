import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.yolo import Yolo
from loss.yolo_loss import YoloLoss
from dataset.voc_dataset import VOCDataset
from torch.utils.data import Subset
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

dataset=VOCDataset(image_set="train")

indices = [4,5,6,7]
single_img_dataset=Subset(dataset,indices)

loader = DataLoader(
    single_img_dataset,
    batch_size=4,
    shuffle=False
)

model=Yolo()
model = Yolo()
criterion=YoloLoss()
optimizer=optim.SGD(
    model.parameters(),
    lr=1e-5,
    momentum=0.0,
    weight_decay=5e-4
)

model.train()

weights_dir = os.path.join(project_root, "weights")
os.makedirs(weights_dir, exist_ok=True)

best_loss=1000000

for step in range(1000):
    for img,target in loader:
        preds=model(img)
        loss=criterion(preds,target)

        optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if loss.item()<best_loss:
            best_loss=loss.item()
            torch.save(
            model.state_dict(),
                os.path.join(weights_dir, "yolov4_8img.pth")
            )

    print(f"Step {step} | Loss: {loss.item():.4f}")


