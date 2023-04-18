import torch
import numpy as np
from model import Net_rand,Net_detection
import torch.nn as nn
import time
def in_model():
    G = Net_rand()
    G.load_state_dict(torch.load("D:\\Project\\Python\\Neroset_PassGan\\Gmodel1.pth"))
    G.eval()
    D=Net_detection()
    D.load_state_dict(torch.load("D:\\Project\\Python\\Neroset_PassGan\\Dmodel1.pth"))
    D.eval()
    return G,D


def save(G,D):
    torch.save(G.state_dict(), "D:\\Project\\Python\\Neroset_PassGan\\Gmodel1.pth")
    torch.save(D.state_dict(), "D:\\Project\\Python\\Neroset_PassGan\\Dmodel1.pth")
def Run():
    dev='cuda:0'
    G=Net_rand()
    D=Net_detection()
    G.to(dev)
    D.to(dev)
    Dcriterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    Gcriterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    # optimizer = torch.optim.Adam(net.parameters(), lr=net.learning_rate)
    Goptimizer = torch.optim.Adam(G.parameters())
    Doptimizer = torch.optim.Adam(D.parameters())
    epoch_kol=1
    y_train=Trainy(G.out())
    y_train=y_train.to(dev)
    Gdop=torch.ones([np.ones(G.out()) .shape[0],1])
    Gdop=Gdop.to(dev)
    Ddopfalse=torch.tensor(np.array([0]).astype(np.float32))
    Ddopfalse=Ddopfalse.to(dev)
    Ddoptrue=torch.tensor(np.array([1]).astype(np.float32))
    Ddoptrue=Ddoptrue.to(dev)
    # ----------------------------------------------------------------------
    for epoch in range(epoch_kol):
        # for i, (images, labels) in enumerate(train_loader):  # Загрузка партии изображений с индексом, данными,
        # классом
        loss = 0
        x_train = Trainx(G.inp())
        x_train=x_train.to(dev)
        # Gloss=[]
        for i in range(len(x_train)):
            # images = Variable(images.view(-1, 28 * 28))  # Конвертация тензора в переменную: изменяем изображение с
            # вектора, размером 784 на матрицу 28 x 28 labels = Variable(labels)
            # Goptimizer.zero_grad()
            Goutputs = G(x_train[i])
            Doutputs=D(Goutputs)
            Dloss=Dcriterion(Doutputs,Ddopfalse)
            #-----------------------
            Doptimizer.zero_grad()
            Dloss.backward()
            Doptimizer.step()
            #----------------------------
            # очень самнительно Gdop. Мы считаем ошибку, как будто выход G должен состоять из 1
            # Glossone=Gcriterion(Goutputs,Gdop)
            # Gloss.append(Glossone)
        # for i in range(len(Gloss)):
        #     Goptimizer.zero_grad()
        #     Gloss[i].backward()
        #     Goptimizer.step()
        for i in range(len(y_train)):
            Doutputs=D(y_train[i])
            Dloss=Dcriterion(Doutputs,Ddoptrue)
            #-----------------------
            Doptimizer.zero_grad()
            Dloss.backward()
            Doptimizer.step()
            #----------------------------
        for i in range(len(x_train)):
            Goutputs = G(x_train[i])
            Doutputs=D(Goutputs)
            Gloss=Gcriterion(Doutputs,Ddoptrue)
            #-----------------------
            Goptimizer.zero_grad()
            Gloss.backward()
            Goptimizer.step()
            #----------------------------
        # print(loss.data.text)
    save(G,D)
def Trainx(kol):
    batch=1
    train_x=torch.from_numpy(np.random.normal(0,1,size=(batch,kol)).astype(np.float32))
    return train_x
def Trainy(kol):
    f=open("10_million_password_list_top_1000000.txt.txt",'r')
    l=list(map(str,f.read().split()))
    l=l[:100000]
    f.close()
    train_y=[]
    for i in range(len(l)):
        if l[i][-1]=='\n':
            l[i]=l[i][:-1]
        p=[]
        for j in l[i]:
           p.append(ord(j)/256)
        if kol > len(l[i]):
            for j in range(kol - len(l[i])):
                p.append(-10)
        if (kol<len(l[i])):
            print(f"len password!: {len(l[i])}")
            exit()
        train_y.append(p)
    train_y=np.array(train_y)
    train_y=train_y.astype(np.float32)
    train_y=torch.from_numpy(train_y)
    return train_y
def print_hi(name):
    tim=time.time()
    Run()
    print(time.time()-tim)
if __name__ == '__main__':
    print_hi('PyCharm')
