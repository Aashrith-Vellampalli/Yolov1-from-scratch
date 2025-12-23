import os
import sys

# Ensure project root is on sys.path when this script is executed directly so
# sibling packages like `model`, `dataset`, and `utils` can be imported.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.yolo import Yolo
import torch
from dataset.voc_dataset import VOCDataset
from utils.decode_yolo import decode_actual,decode_pred
from PIL import Image, ImageDraw
import torchvision.transforms as T

def fix_box(x1, y1, x2, y2, img_size=448):
    x_min = max(0, min(x1, x2))
    y_min = max(0, min(y1, y2))
    x_max = min(img_size, max(x1, x2))
    y_max = min(img_size, max(y1, y2))
    return x_min, y_min, x_max, y_max


model = Yolo()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model.load_state_dict(
    torch.load(os.path.join(project_root, "weights", "yolov4_8img.pth"),
               map_location="cpu")
)

model.eval()
dataset=VOCDataset(image_set="train")

os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
to_pil=T.ToPILImage()

for i in range(4):
    img,target=dataset[i+4]
    img_input=img.unsqueeze(0)
    img_vis=img

    with torch.no_grad():
        pred=model(img_input)
        boxes_pred=decode_pred(pred)
        boxes_actual=decode_actual(target)
    
    img_pil=to_pil(img_vis)
    draw=ImageDraw.Draw(img_pil)

    for box in boxes_actual:
        x1,y1,x2,y2,class_id=box
        x1, y1, x2, y2 = fix_box(x1, y1, x2, y2)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
    
    for box in boxes_pred:
        x1,y1,x2,y2,conf,class_id=box
        x1, y1, x2, y2 = fix_box(x1, y1, x2, y2)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    save_path = os.path.join(project_root, "outputs", f"img_{i}.png")
    img_pil.save(save_path)
    print("saved image",i+1)


