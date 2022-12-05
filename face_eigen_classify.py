import cv2
from numpy import *
import numpy as np
import torch
import torch.nn as nn


def pca():
    img_list = []
    for i in range(1,41):
        for j in range(1,11):
            file = "./orl_faces/s%d/%d.bmp"%(i,j)
            img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32)/255.0
            img_list.append(img)
    imgs = np.zeros([400,10304],dtype=np.float32)

    for i in range(0,400):
        imgs[i,:] = img_list[i].reshape(-1)
    
    #imgs [400,10304]

    imgs_mean = imgs.sum(axis=0)/400.0
    np.savetxt('mean.txt',imgs_mean,fmt="%.5f")
    imgs = imgs - imgs_mean

    10304*10304
    conv = imgs.transpose(1,0).dot(imgs)
    eig_value,eig_vector = np.linalg.eig(conv)
    eig_value = eig_value.astype(dtype = np.float32)
    eig_vector = eig_vector.astype(dtype = np.float32)
    np.savetxt('eig_vector.txt',eig_vector,fmt="%.5f")
    np.savetxt('eig_value.txt',eig_value,fmt="%.5f")
    eig_vector.tofile("eig_vector.bin")

    #400*400
    # conv = imgs.dot(imgs.transpose(1,0))
    # eig_value,eig_vector = np.linalg.eig(conv)
    # eig_value = eig_value.astype(dtype = np.float32)
    # eig_vector = eig_vector.astype(dtype = np.float32)
    # eig_vector = imgs.transpose(1,0).dot(eig_vector)
    # np.savetxt('eig_vector.txt',eig_vector,fmt="%.5f")
    # np.savetxt('eig_value.txt',eig_value,fmt="%.5f")
    # eig_vector.tofile("eig_vector.bin")


def feature_extraction():
    imgs_mean = np.loadtxt("mean.txt")
    eig_vector = np.fromfile('eig_vector.bin',dtype = np.float32)
    eig_vector = eig_vector.reshape(10304,-1)
    u = eig_vector[:,:256]
    for i in range(0,256):
        face = u[:,i] + imgs_mean
        face = face * 255
        face[face<0] = 0
        face[face>255] = 255
        face = face.astype(np.uint8).reshape(112,-1)
        cv2.imwrite('eigface/%s.bmp'%(i+1),face)


    feats = np.zeros([400,256])
    for i in range(1,41):
        for j in range(1,11):
            file = "orl_faces/s%d/%d.bmp"%(i,j)
            img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32)/255.0
            img = img.reshape(-1)
            img = img - imgs_mean
            prj = img.dot(u)
            feats[(i-1)*10+j-1,:] = prj

            #画出提取的特征还原的图像
            cons = u.dot(prj)
            cons = cons + imgs_mean
            cons = cons*255
            cons[cons<0] = 0
            cons[cons>255]=255
            cons = cons.astype(np.uint8).reshape(112,-1)
            cv2.imwrite('construct/%s_%s..bmp'%(i,j),cons)
    return feats

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.linear1 = nn.Linear(256,512)
        self.linear2 = nn.Linear(512,40)
    def forward(self,x):
        x = self.linear1(x)
        x = self.linear2(x)

        return x


if __name__ == "__main__":
    # pca()
    feats = feature_extraction()
    
    train_x = np.zeros([360,256])
    train_y = np.zeros([360],dtype = np.int32)
    for i in range(0,40):
        for j in range(0,9):
            train_x[9*i+j,:] = feats[i*10+j,:]
            train_y[9*i+j] = i

    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).long()
    shuffle_index = torch.randperm(360)
    train_x = train_x[shuffle_index]
    train_y = train_y[shuffle_index]

    model = MLP()
    loss_f = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    for epoch in range(1,401):
        losses = []
        for batch_idx in range(18):
            batch_x = train_x[batch_idx*20:(batch_idx+1)*20,:]
            batch_y = train_y[batch_idx*20:(batch_idx+1)*20]
            y = model(batch_x)
            loss = loss_f(y,batch_y)
            losses.append(float(loss))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if(epoch%10==0):
            print('第'+str(epoch)+'个epoch的loss值：'+str(float(mean(losses))))
            

    eval_x = np.zeros([40,256])
    for c in range(0,40):
        eval_x[c,:] = feats[c*10+9,:]
    eval_x = torch.from_numpy(eval_x).float()
    eval_y = model(eval_x)
    eval_y = eval_y.softmax(dim=1)
    eval_y = eval_y.argmax(dim=1)
    print(eval_y)

    cnt = 0
    for i in range(eval_y.shape[0]):
        if i == eval_y[i]:
            cnt+=1
    
    print('precision : %.2f%%' % (cnt/40.0 * 100))