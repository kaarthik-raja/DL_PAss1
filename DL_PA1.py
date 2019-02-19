#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import argparse as agp


# In[11]:


import os 
train_path=os.path.join("Data","train.csv")
test_path=os.path.join("Data", "test.csv")
valid_path=os.path.join("Data", "valid.csv")


# In[12]:


train=pd.read_csv(train_path)
test=pd.read_csv(test_path)
valid=pd.read_csv(valid_path)

#create global variables
lr= 0.01 # learning rate
momentum=0.5 
num_hidden=3 
sizes=np.zeros(3)
activation= "sigmoid"
lossfn= "sq" 
opt= "adam" # optimizer
batch_size=20 
epochs=5
anneal=True
#save paths
save_dir=""
train=""
val=""
test=""

# In[18]:


train=np.asarray(train)
test=np.asarray(test)


# In[65]:


#np.random.shuffle(train)
# np.random.shuffle(test)
sizes=np.array([10,15,20])
np.append(sizes,10)


# In[114]:


#initialize weights and bias
sizes=np.array([10,15,20])
num_hidden=3
sizes=np.asarray(sizes)
sizes=np.insert(sizes,0,784)
sizes=np.append(sizes,10)
print(sizes)
wt=[] #list of weight matrices 
bias=[] #list of bias vectors
for n in range(num_hidden):
    w=np.random.rand(sizes[n],sizes[n+1])
    b=np.random.rand(1,sizes[n+1])  
    wt.append(w)
    bias.append(b)
    
w=np.random.rand(sizes[num_hidden],10)
b=np.random.rand(10)  
wt.append(w)
bias.append(b)

# def fgrad(hs):
    # pass

# def fval(a):
    # pass

# def outputError(y,oneH):
    # pass



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
    Dak = yhat -  oneH
    #print(sizes)
    prev_w=0
    prev_b=0
    for k in range(num_hidden,-1,-1):
        #print(k,"=k",hs[k].shape,Dak.shape)
        Dwk= np.zeros((sizes[k],sizes[k+1]))
        #print(Dwk.shape,hs[k][0].shape,Dak[0].shape,np.outer(hs[k][0],Dak[0]).shape)
        for i in range(train.shape[0]-1):
            Dwk = np.add(Dwk,np.outer(hs[k][i],Dak[i]))

        wt[k] = wt[k] - eta * np.divide(Dwk,train.shape[0]-1) - gamma*prev_w
        #print("Dwk",Dwk)
        Dbk = Dak
        bias[k] = bias[k] - eta * np.mean(Dbk,axis=0) - gamma*prev_b
        prev_w += np.divide(Dwk,train.shape[0]-1)
        prev_b += np.mean(Dbk, axis=0)
        Dhk = np.matmul(Dak , np.transpose(wt[k]))
        Dak = np.multiply(Dhk,np.multiply(hs[k] , 1-hs[k]) )        
    
    return yhat




# for i in range(10):
#     yc=vanilla_grad_desc(num_hidden,sizes)


# In[71]:
gamma=0.9
for i in range(3):
    ycm=momentum_grad_desc(num_hidden,sizes,momentum,gamma)
train[:,785]


# In[131]:


a=np.array([1,2,3,4,0])
b=np.arange(5)
oneH = np.zeros((5,6))

oneH[b,a]=1
oneH
if type(y[0]) is int:
    
    print( "hi")
print(y[0])


# In[129]:


np.shape(y)


def main():
    pass
        
if __name__=="__main__":
    main()
    







