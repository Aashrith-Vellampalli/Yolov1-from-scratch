import torch
import torch.nn as nn

S=7
B=2
C=20

class Yolo(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,stride=2,kernel_size=7,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1,inplace=True),
            
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv_block2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1,inplace=True),
            
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv_block3=nn.Sequential(
            nn.Conv2d(in_channels=192,out_channels=128,kernel_size=1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1,inplace=True),
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,inplace=True),
            
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,inplace=True),
           
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,inplace=True),
            
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        conv_block4=[]
        for _ in range(4):
            conv_block4.append(nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,bias=False))
            conv_block4.append(nn.BatchNorm2d(256))
            conv_block4.append(nn.LeakyReLU(0.1,inplace=True))
            
            conv_block4.append(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,bias=False))
            conv_block4.append(nn.BatchNorm2d(512))
            conv_block4.append(nn.LeakyReLU(0.1,inplace=True))
        
        conv_block4.append(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1,bias=False))
        conv_block4.append(nn.BatchNorm2d(512))
        conv_block4.append(nn.LeakyReLU(0.1,inplace=True))

        conv_block4.append(nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1,bias=False))
        conv_block4.append(nn.BatchNorm2d(1024))
        conv_block4.append(nn.LeakyReLU(0.1,inplace=True))

        conv_block4.append(nn.MaxPool2d(kernel_size=2,stride=2))

        self.conv_block4=nn.Sequential(*conv_block4)

        conv_block5=[]
        for _ in range(2):
            conv_block5.append(nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1,bias=False))
            conv_block5.append(nn.BatchNorm2d(512))
            conv_block5.append(nn.LeakyReLU(0.1,inplace=True))
            
            conv_block5.append(nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1,bias=False))
            conv_block5.append(nn.BatchNorm2d(1024))
            conv_block5.append(nn.LeakyReLU(0.1,inplace=True))

        conv_block5.append(nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1,bias=False))
        conv_block5.append(nn.BatchNorm2d(1024))
        conv_block5.append(nn.LeakyReLU(0.1,inplace=True))

        conv_block5.append(nn.MaxPool2d(kernel_size=2,stride=2))

        self.conv_block5=nn.Sequential(*conv_block5)

        self.conv_block6=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1,inplace=True),

            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1,inplace=True),
        )

        self.fc=nn.Sequential(
            nn.Linear(in_features=1024*7*7,out_features=4096),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features=4096,out_features=S*S*(B*5+C))
        )
    
    def forward(self,x):
        x= self.conv_block1(x)
        x= self.conv_block2(x)
        x= self.conv_block3(x)
        x= self.conv_block4(x)
        x= self.conv_block5(x)
        x= self.conv_block6(x)

        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        x=x.view(x.shape[0],S,S,(B*5)+C)

        return x




        



        