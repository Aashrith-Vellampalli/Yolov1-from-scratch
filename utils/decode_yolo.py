import torch

def decode_pred(pred,conf_threshold=0.3,img_size=448):
    pred=pred.squeeze(0)
    box_cords=[]
    cell_size=img_size/7
    for i in range(7):
        for j in range(7):
            cell=pred[i,j]

            b1=cell[0:5]
            b2=cell[5:10]

            if(b1[4]>b2[4]):
                box=b1
            else:
                box=b2
            
            if(box[4]<conf_threshold):
                continue
            
            conf=box[4]
            
            class_prob=cell[10:]
            class_id=torch.argmax(class_prob,dim=0).item()
            x_cell,y_cell,w,h=box[0],box[1],box[2],box[3]

            x_center= (j+x_cell)*cell_size
            y_center= (i+y_cell)*cell_size
            w_img=w*img_size
            h_img=h*img_size

            x1=x_center-w_img/2
            x2=x_center+w_img/2
            y1=y_center-h_img/2
            y2=y_center+h_img/2

            box_cords.append((x1.item(),y1.item(),x2.item(),y2.item(),conf.item(),class_id))
    
    return box_cords


def decode_actual(target,img_size=448):
    cell_size=img_size/7
    box_cords=[]
    
    for i in range(7):
        for j in range(7):
            if target[i,j,4]==1:
                cell=target[i,j]
                class_prob=cell[5:]
                class_id=torch.argmax(class_prob,dim=0).item()
                x_cell,y_cell,w,h=cell[0],cell[1],cell[2],cell[3]

                x_center= (j+x_cell)*cell_size
                y_center= (i+y_cell)*cell_size
                w_img=w*img_size
                h_img=h*img_size

                x1=x_center-w_img/2
                x2=x_center+w_img/2
                y1=y_center-h_img/2
                y2=y_center+h_img/2

                box_cords.append((x1.item(),y1.item(),x2.item(),y2.item(),class_id))

    return box_cords


