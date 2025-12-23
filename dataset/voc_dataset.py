import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import torchvision.transforms as T

VOC_CLASSES = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19
}

class VOCDataset(Dataset):
    def __init__(self,image_set="train",img_size=448):
        self.img_size=img_size

        self.voc=VOCDetection(
            root="data",
            year="2007",
            image_set=image_set,
            download=False
        )

        self.transform=T.Compose([
            T.Resize((img_size,img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.voc)
    
    def __getitem__(self, idx):
        img,target=self.voc[idx]
        img=self.transform(img)

        annotation=target["annotation"]
        objects=annotation["object"]

        if not isinstance (objects,list):
            objects=[objects]

        target_tensor=torch.zeros(7,7,25)
        
        img_w=float(annotation["size"]["width"])
        img_h=float(annotation["size"]["height"])
        
        for obj in objects:
            class_name=obj["name"]
            box=obj["bndbox"]

            
            xmin=float(box["xmin"])
            ymin=float(box["ymin"])
            xmax=float(box["xmax"])
            ymax=float(box["ymax"])

            x_center=(xmin+xmax)/2.0
            x_center=x_center/img_w
            y_center=(ymin+ymax)/2.0
            y_center=y_center/img_h

            xcord=int(x_center*7)
            ycord=int(y_center*7)
            xcord=min(xcord,6)
            ycord=min(ycord,6)

            if target_tensor[ycord][xcord][4]==1:
                continue
            
            w=(xmax-xmin)/img_w
            h=(ymax-ymin)/img_h
            x_rel=x_center*7-xcord
            y_rel=y_center*7-ycord

            target_tensor[ycord][xcord][0]=x_rel
            target_tensor[ycord][xcord][1]=y_rel
            target_tensor[ycord][xcord][2]=w
            target_tensor[ycord][xcord][3]=h
            target_tensor[ycord][xcord][4]=1
            class_id=VOC_CLASSES[class_name]
            target_tensor[ycord,xcord,5+class_id]=1
        
        return img,target_tensor
    


            

            


