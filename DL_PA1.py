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

gamma=0.9
eta=10
adam_b1=0.9
adam_bp1=0.9
adam_b2=0.999
adam_bp2=0.999
adam_epsilon= 3
#initialize weights and bias
wt=[] #list of weight matrices 
bias=[] #list of bias vectors

momentum_w=[]
momentum_b=[]


look_w=[]
look_b=[]
update_w=[]
update_b=[]

adam_w_m=[]
adam_w_v=[]
adam_b_m=[]
adam_b_v=[]
def initwts():
    global sizes
    sizes=np.asarray(sizes)
    sizes=np.insert(sizes,0,784)
    sizes=np.append(sizes,10)

    for n in range(num_hidden+1):
        w=np.divide(np.subtract(np.random.rand(sizes[n],sizes[n+1]),0.8),1000000)
        b=np.divide(np.subtract(np.random.rand(sizes[n+1]),0.5),100)  
        
        wt.append(w)
        bias.append(b)
        if opt =="momentum":
            momentum_w.append(np.zeros((sizes[n],sizes[n+1])))
            momentum_b.append(np.zeros((sizes[n+1])))
        elif opt == "nag":
            look_w.append(w)
            look_b.append(b)
            update_w.append(np.zeros((sizes[n],sizes[n+1])))
            update_b.append(np.zeros((sizes[n+1])))
        elif opt == "adam":
            adam_w_m.append(np.zeros((sizes[n],sizes[n+1])))
            adam_w_v.append(np.zeros((sizes[n],sizes[n+1])))
            adam_b_m.append(np.zeros((sizes[n+1])))
            adam_b_v.append(np.zeros((sizes[n+1])))

def fgrad(hs):
    if activation == "sigmoid":
        return np.multiply(hs,1-hs)
    else:
        return np.subtract(1,np.multiply(hs , hs))

def fval(a):
    if activation == "sigmoid":
        return np.reciprocal(np.add(1,np.exp(-a)))
    else:
        return np.multiply(np.subtract(np.exp(a),np.exp(-a)),np.reciprocal( np.add(np.exp(a),np.exp(-a)) ) )

# def outputError(y,oneH):
    # pass
def optimizer(k,Dwk,Dbk):
    global wt,bias,momentum_w,momentum_b
    print("optimizer",opt,k,Dwk[1,1:4])
    if opt == "gd":
        wt[k]=np.subtract(wt[k],np.multiply(eta,Dwk))
        bias[k]=np.subtract(bias[k],np.multiply(eta,Dbk))
    elif opt == "momentum":
        momentum_w[k]=np.multiply(momentum_w[k],gamma)
        momentum_w[k]=np.add(momentum_w[k],np.multiply(eta,Dwk))
        wt[k]=np.subtract(wt[k],momentum_w[k])        
        momentum_b[k]=np.multiply(momentum_b[k],gamma)
        momentum_b[k]=np.add(momentum_b[k], np.multiply(eta,Dbk))
        bias[k]=np.subtract(bias[k], momentum_b[k])
    elif opt == "nag":
        update_w[k]= np.sum( np.multiply(gamma,update_w[k]),np.multiply(eta,Dwk) )
        look_w[k]= np.subtract(look_w[k],update_w[k])
        update_b[k]= np.sum( np.multiply(gamma,update_b[k]),np.multiply(eta,Dbk) )
        look_b[k]= np.subtract(look_b[k],update_b[k])
    elif opt == "adam":
        adam_w_m = np.add(np.multiply(adam_b1,adam_w_m),np.multiply(1-adam_b1,Dwk))
        adam_b_m = np.add(np.multiply(adam_b1,adam_b_m),np.multiply(1-adam_b1,Dbk))
        
        adam_w_v = np.add(np.multiply(adam_b2,adam_w_v),np.multiply(1-adam_b2,np.multiply(Dwk,Dwk)))
        adam_b_v = np.add(np.multiply(adam_b2,adam_b_v),np.multiply(1-adam_b2,np.multiply(Dbk,Dbk)))
        
        wt[k] = np.subtract(wt[k],np.multiply( np.multiply(eta, np.reciprocal( np.sqrt( np.add( np.divide(adam_w_v,1-adam_bp2) ,adam_epsilon) )) ), np.divide(adam_w_m,1-adam_bp1) ) )
        bias[k] = np.subtract(bias[k],np.multiply( np.multiply(eta, np.reciprocal( np.sqrt( np.add( np.divide(adam_b_v,1-adam_bp2) ,adam_epsilon) )) ), np.divide(adam_b_m,1-adam_bp1) ) )
        

# implementing functions to do different tasks. This is the main function block
#def vanilla_grad_desc(num_hidden,sizes):

# # implementing functions to do different tasks. This is the main function block
# def vanilla_grad_desc(num_hidden,sizes):
#     eta=0.1
#     x=train[1:,1:785]
#     y=train[1:,785]
#     hs=[]
#     # calculate the initial class out put from w and b initializations
#     #for im in range(x.size[0]):
    
#     h=x
#     hs.append(x)
#     for n in range(num_hidden):
#             a=np.add(np.matmul(h,wt[n]),bias[n])
#             h=np.reciprocal(np.add(1,np.exp(-a)))
#             hs.append(h)
#     a=np.add(np.matmul(h,wt[num_hidden]),bias[num_hidden]) 
    
#     yhat= np.divide(np.exp(a),np.sum(np.exp(a)))
    
#     #for CEEF
#     oneH = np.zeros((x.shape[0],10))
    
#     oneH[np.arange(x.shape[0]),y.astype(int)]=1
    
#     loss =  -np.mean(np.multiply( np.log(yhat) ,oneH))
#     print("loss",loss)
#     Dak = yhat -  oneH
#     #print(sizes)
#     for k in range(num_hidden,-1,-1):
#         #print(k,"=k",hs[k].shape,Dak.shape)
#         Dwk= np.zeros((sizes[k],sizes[k+1]))
#         #print(Dwk.shape,hs[k][0].shape,Dak[0].shape,np.outer(hs[k][0],Dak[0]).shape)
#         for i in range(train.shape[0]-1):
#             Dwk = np.add(Dwk,np.outer(hs[k][i],Dak[i]))
#         wt[k] = wt[k] - eta * np.divide(Dwk,train.shape[0]-1)
#         #print("Dwk",Dwk)
#         Dbk = Dak
#         bias[k] = bias[k] - eta * np.mean(Dbk,axis=0)
#         Dhk = np.matmul(Dak , np.transpose(wt[k]))
#         Dak = np.multiply(Dhk,np.multiply(hs[k] , 1-hs[k]) )        
    
#     return yhat

def momentum_grad_desc(num_hidden,sizes,momentum,gamma):
    eta=0.1
    x=train[1:,1:785]
    y=train[1:,785]
    hs=[]
    # calculate the initial class out put from w and b initializations
    #for im in range(x.size[0]):
    
    h=x
    hs.append(x)
    
    #if opt == "nag":
    #    wt[k] = np.subtract(look_w[k],np.multiply(gamma,update_w[k]) )
    #    bias[k] = np.subtract(look_b[k],np.multiply(gamma,update_b[k]) )

    #forward Propagation
    for n in range(num_hidden):
            a=np.add(np.matmul(h,wt[n]),bias[n])
            h=fval(a)
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
        Dwk= np.zeros((sizes[k],sizes[k+1]))
        for i in range(train.shape[0]-1):
            Dwk = np.add(Dwk,np.outer(hs[k][i],Dak[i]))
        # print("hs_Dak_Dwk",hs[k].shape,Dak.shape,Dwk.shape)
        # print("hs_Dak_Dwk",hs[k][4,10:20],Dak[1,1:4],Dwk[1:4,1:4])


        Dwk = np.divide(Dwk,train.shape[0]-1)
        Dbk = Dak
        Dbk = np.mean(Dbk,axis=0)
        optimizer(k,Dwk,Dbk)
        Dhk = np.matmul(Dak , np.transpose(wt[k]))
        # print("Dhk",Dhk[1,1:4])

        Dak = np.multiply(Dhk,  fgrad(hs[k]) )       
        
        print("fgrad",fgrad(hs[k])[1,1:4])
        # print("Dak",Dak[1,1:4])
    
    if opt == "adam":
        adam_bp1=adam_bp1*adam_b1
        adam_bp2=adam_bp2*adam_b2
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
    print(train.values)
    test=pd.read_csv(test_path)
    valid=pd.read_csv(valid_path)

    train=np.asarray(train)
    test=np.asarray(test)
    print("sizes",sizes)
    #np.random.shuffle(train)
    # np.random.shuffle(test)

    initwts()
    gamma=0.9
    for i in range(3):
        ycm=momentum_grad_desc(num_hidden,sizes,momentum,gamma)
    # for i in range(10):
        # yc=vanilla_grad_desc(num_hidden,sizes)


if __name__=="__main__":
    main()
    







