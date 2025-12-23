import torch
"""
predicted would be (B,4)
actual would be (B,4)

they would be of the form (x_center,y_center,W,H)
"""
def iou(predicted,actual):
    box_right=predicted[...,0:1]+predicted[...,2:3]/2
    box_up=predicted[...,1:2]+predicted[...,3:4]/2
    box_left=predicted[...,0:1]-predicted[...,2:3]/2
    box_down=predicted[...,1:2]-predicted[...,3:4]/2

    actual_right=actual[...,0:1]+actual[...,2:3]/2
    actual_up=actual[...,1:2]+actual[...,3:4]/2
    actual_left=actual[...,0:1]-actual[...,2:3]/2
    actual_down=actual[...,1:2]-actual[...,3:4]/2

    x1=torch.min(box_right,actual_right)
    y1=torch.min(box_up,actual_up)
    x2=torch.max(box_left,actual_left)
    y2=torch.max(box_down,actual_down)

    intersection = (x1 - x2).clamp(min=0) * (y1 - y2).clamp(min=0)
    
    pred_area=(box_right-box_left)*(box_up-box_down)
    actual_area=(actual_right-actual_left)*(actual_up-actual_down)

    return intersection/(pred_area+actual_area-intersection+1e-6)



