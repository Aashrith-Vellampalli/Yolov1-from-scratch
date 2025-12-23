import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from utils.iou import iou
except ModuleNotFoundError:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils.iou import iou

"""
predictions is (B,7,7,30) where there are 2 bounding boxes 
target is (B,7,7,25) where there is one bounding 
"""

class YoloLoss(nn.Module):
    def __init__(self,S=7,B=2,C=20):
        
        super().__init__()
        self.S=S
        self.B=B
        self.C=C
        self.lambda_coord=5
        self.lambda_noobj=0.5

    def forward(self,predictions,target):
        exist_obj=target[...,4:5]
        
        iou_b1=iou(predictions[...,0:4]*exist_obj,target[...,0:4]*exist_obj)
        iou_b2=iou(predictions[...,5:9]*exist_obj,target[...,0:4]*exist_obj)
        
        bestbox=(iou_b2>iou_b1).float()


        """
        box loss
        """
        pred_box_xy=bestbox*predictions[...,5:7]+(1-bestbox)*predictions[...,0:2]
        actual_box_xy=target[...,0:2]
        
        pred_box_wh = torch.sqrt(
            torch.clamp(
            bestbox*predictions[...,7:9] +
            (1-bestbox)*predictions[...,2:4],
            min=1e-6
            )
        )

        actual_box_wh = torch.sqrt(
        torch.clamp(
            target[...,2:4],
            min=1e-6
        )
        )



        box_xy_loss=F.mse_loss(
            exist_obj*pred_box_xy,
            exist_obj*actual_box_xy,
            reduction="sum"
        )
        
        box_wh_loss=F.mse_loss(
            exist_obj*pred_box_wh,
            exist_obj*actual_box_wh,
            reduction="sum"
        )


        total_box_loss=self.lambda_coord*(box_xy_loss+box_wh_loss)

        """
        confidence loss
        target_confidence=p(obj)*(IOU(pred,actual)) whichever box is max
        """
        pred_box_con=bestbox*predictions[...,9:10]+(1-bestbox)*predictions[...,4:5]
        actual_box_con=bestbox*iou_b2+(1-bestbox)*iou_b1

        obj_con_loss=F.mse_loss(
            exist_obj*pred_box_con,
            exist_obj*actual_box_con,
            reduction="sum"
        )
        
        noobj_con_loss=self.lambda_noobj*(
        F.mse_loss(
            (1-exist_obj)*predictions[...,4:5],
            torch.zeros_like(predictions[...,4:5]),
            reduction="sum"
        )+
        F.mse_loss(
            (1-exist_obj)*predictions[...,9:10],
            torch.zeros_like(predictions[...,9:10]),
            reduction="sum"
        )
        )

        total_con_loss=obj_con_loss+noobj_con_loss

        """
        class loss
        """
        target_prob=target[...,5:25]
        predict_prob=predictions[...,10:30]

        total_prob_loss=F.mse_loss(
            exist_obj*predict_prob,
            exist_obj*target_prob,
            reduction="sum"
        )

        """
        total loss = sum of all 3
        """

        loss=total_box_loss+total_con_loss+total_prob_loss
        
        return loss




    


