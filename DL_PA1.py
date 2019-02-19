#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import argparse as agp
import os 



#create global variables
# lr= 0.01 # learning rate
# momentum=0.5 
# num_hidden=3 
# sizes=np.zeros(3)
# activation= "sigmoid"
# lossfn= "sq" 
# opt= "adam" # optimizer
# batch_size=20 
# epochs=5
# anneal=True

#save paths
# save_dir=""
# train_path=""
# valid_path=""
# test_path=""




#initialize weights and bias
wt=[] #list of weight matrices 
bias=[] #list of bias vectors

def initwts():
    global sizes
    sizes=np.asarray(sizes)
    sizes=np.insert(sizes,0,784)
    sizes=np.append(sizes,10)

    for n in range(num_hidden):
        w=np.random.rand(sizes[n],sizes[n+1])
        b=np.random.rand(1,sizes[n+1])  
        wt.append(w)
        bias.append(b)
        
    w=np.random.rand(sizes[num_hidden],10)
    b=np.random.rand(10)  
    wt.append(w)
    bias.append(b)


def fgrad(hs):
    pass

def fval(a):
    pass

def outputError(y,oneH):
    pass


# implementing functions to do different tasks. This is the main function block
def vanilla_grad_desc(num_hidden,sizes):
    eta=0.1
    x=train[1:,1:785]
    y=train[1:,785]
    hs=[]
    # calculate the initial class out put from w and b initializations
    #for im in range(x.size[0]):
    
    h=x

    #forward Propagation
    hs.append(x)
    for n in range(num_hidden):
            a=np.add(np.matmul(h,wt[n]),bias[n])
            h=np.reciprocal(np.add(1,np.exp(-a)))
            hs.append(h)
    a=np.add(np.matmul(h,wt[num_hidden]),bias[num_hidden]) 
    
    yhat= np.divide(np.exp(a),np.sum(np.exp(a)))
    
    #for CEEF
    oneH = np.zeros((x.shape[0],10))
    
    oneH[np.arange(x.shape[0]),y.astype(int)]=1
    
    loss =  -np.mean(np.multiply( np.log(yhat) ,oneH))
    print("loss",loss)
    

    #backward Propagation
    Dak = yhat -  oneH
    #print(sizes)

    for k in range(num_hidden,-1,-1):
        #print(k,"=k",hs[k].shape,Dak.shape)
        Dwk= np.zeros((sizes[k],sizes[k+1]))
        #print(Dwk.shape,hs[k][0].shape,Dak[0].shape,np.outer(hs[k][0],Dak[0]).shape)
        for i in range(train.shape[0]-1):
            Dwk = np.add(Dwk,np.outer(hs[k][i],Dak[i]))
        wt[k] = wt[k] - eta * np.divide(Dwk,train.shape[0]-1)
        #print("Dwk",Dwk)
        Dbk = Dak
        bias[k] = bias[k] - eta * np.mean(Dbk,axis=0)
        Dhk = np.matmul(Dak , np.transpose(wt[k]))
        Dak = np.multiply(Dhk,np.multiply(hs[k] , 1-hs[k]) )        

    return yhat
    
def csv_list(string):
   return [ int(i) for i in string.split(',')]

def annealf(string):
    if string in ["true","True","T","t","1" ] :
        return True
    elif string in ["False","false","F","f","0"]:
        return False

def main():
    global lr,momentum,num_hidden,sizes,activation,loss,opt,batch_size,epoch,anneal,save_dir,expt_dir,train_path,test_path,valid_path
    global train,test,valid
    print("parsing...")
    parser = agp.ArgumentParser()
    parser.add_argument("--lr", type=float, help="the learning rate", default=0.01)
    parser.add_argument("--momentum", type=float, help="the momentum in lr", default=0.5)
    parser.add_argument("--num_hidden", type=int, help="# of Hidden Layers", default= 3)
    parser.add_argument("--sizes", type=csv_list, help="# of Nodes per H_Layer", default= [100,100,100])
    parser.add_argument("--activation", type=str, help="activation function", default= "sigmoid", choices=["sigmoid","tanh"])
    parser.add_argument("--loss", type=str, help="loss function", default= "ce", choices=["sq","ce"])
    parser.add_argument("--opt", type=str, help="optimizer", default= "gd", choices=["gd","momentum","nag","adam"])
    parser.add_argument("--batch_size", type=int, help="batch size per step", default= 20)
    parser.add_argument("--epoch", type=int, help="# of EPOCHs", default= 5)
    parser.add_argument("--anneal", type=annealf, help="anneal", default= True,choices=[True,False])
    parser.add_argument("--save_dir", type=str, help="Save dir location", default= "pa1")
    parser.add_argument("--expt_dir", type=str, help="expt_dir location", default= os.path.join("pa1","exp1"))
    parser.add_argument("--train", type=str, help="train file location", default= os.path.join("Data","train.csv"))
    parser.add_argument("--test", type=str, help="test file location", default= os.path.join("Data","test.csv"))
    parser.add_argument("--validation", type=str, help="validation file location", default= os.path.join("Data","valid.csv"))
    args=parser.parse_args()

    lr,momentum=args.lr,args.momentum
    num_hidden,sizes=args.num_hidden,args.sizes
    activation,loss,opt=args.activation,args.loss,args.opt
    batch_size,epoch=args.batch_size,args.epoch
    anneal=args.anneal
    save_dir,expt_dir=args.save_dir,args.expt_dir
    train_path=args.train
    test_path=args.test
    valid_path=args.validation

    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    valid=pd.read_csv(valid_path)

    train=np.asarray(train)
    test=np.asarray(test)

    #np.random.shuffle(train)
    # np.random.shuffle(test)

    initwts()

    for i in range(10):
        yc=vanilla_grad_desc(num_hidden,sizes)


if __name__=="__main__":
    main()
    







