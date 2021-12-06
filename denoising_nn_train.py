import numpy as np
import torch
import torch.optim as optim
import os
from PIL import Image

class DefenseNetwork(torch.nn.Module):
    def __init__(self):
        super(DefenseNetwork,self).__init__()
        self.encoder=torch.nn.Sequential(
                        torch.nn.Linear(64*64*3,512),
                        torch.nn.ReLU(True),
                        torch.nn.Linear(512,128),
                        torch.nn.ReLU(True),
                        torch.nn.Linear(128,64),
                        torch.nn.ReLU(True))

        self.decoder=torch.nn.Sequential(torch.nn.Linear(64,128),
                        torch.nn.ReLU(True),
                        torch.nn.Linear(128,512),
                        torch.nn.ReLU(True),
                        torch.nn.Linear(512,64*64*3),
                        torch.nn.Sigmoid())

    def forward(self,x):
        x = x.reshape(-1)
        x=self.encoder(x)
        x=self.decoder(x)
        return x.reshape(64,64,3)

dataset="PascalVOC2012"
model = DefenseNetwork()
optimizer=optim.SGD(model.parameters(),lr=0.01,weight_decay=1e-5)
criterion=torch.nn.MSELoss()
losslist=[]
epochloss=0
running_loss=0

for epoch in range(1000):
    count = 0
    for i in os.listdir(dataset):
        if i[0] == ".":
            continue
        imgs = os.listdir(dataset+"/" + i + "/input/")
        gts = os.listdir(dataset+"/" + i + "/groundtruth/")
        imgs.sort()
        gts.sort()
        for j in range(len(imgs)):
            img = Image.open(dataset+'/'+i +'/input/'+imgs[j])
            img = img.resize((64,64))
            img = np.array(img, dtype = np.float)
            gt = np.array(Image.open(dataset+'/'+i +'/groundtruth/'+gts[j]))
            noise=np.random.normal(-5.9,5.9,img.shape)
            img_noise = img + noise
            img_noise = torch.from_numpy(img_noise).float()
            output=model(img_noise)
            img = torch.from_numpy(img).float()
            loss=criterion(output,img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            epochloss+=loss.item()
            count+=1
        losslist.append(running_loss/count)
        running_loss=0
        print("======> epoch: {}/{}, Loss:{}".format(epoch,1000,loss.item()))


torch.save(model.state_dict(), "defense.pt")