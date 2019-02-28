#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import argparse as agp
import os 
from collections import Counter as freq_ 
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import json
import pickle
from hfun import *
 
#set random seed for replicability of results
np.random.seed(1234)
momentum=0.9
adam_b1=0.9
adam_bp1=0.9
adam_b2=0.999
adam_bp2=0.999
adam_epsilon= 0.0000001
ploss=1000000.0
nof_PCA = 100
nofpp =0
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
# def initwts():

# 	global sizes,wt,bias
# 	global momentum_w,momentum_b
# 	global look_w,look_b
# 	global adam_w_m,adam_w_v,adam_b_m,adam_b_v,adam_bp2,adam_bp1
	
# 	sizes=np.asarray(sizes)
# 	sizes=np.insert(sizes,0,nof_PCA)
# 	sizes=np.append(sizes,10).astype(int)

# 	adam_bp1=0.9
# 	adam_bp2=0.999

# 	same_model = False
# 	if os.path.exists(os.path.join("Model4","model.pkl")):
# 		with open(os.path.join("Model4","model.pkl"), 'rb') as f:
# 			model=pickle.load(f)
# 			nsizes  = model["sizes"]
# 			adam_bp2 = model.get("adam_bp2",adam_b2)
# 			adam_bp1 = model.get("adam_bp1",adam_b1)
# 			same_model = True
# 			if num_hidden == model["num_hidden"]:
# 				for k in range(num_hidden+1):
# 					if nsizes[k] != sizes[k]:
# 						same_model = False
# 			else:
# 				same_model = False
# 			print("Same  Model :",same_model)
# 	if same_model:
# 		wt = np.load(os.path.join("Model4","wt.npy"))
# 		bias = np.load(os.path.join("Model4","bias.npy"))
# 		print("continuing previous model data....")
# 	else:
# 		wt= [None]*(num_hidden+1)
# 		bias = [None]*(num_hidden+1)
# 		for n in range(num_hidden+1):
# 			w=np.random.randn(sizes[n],sizes[n+1])/np.sqrt(sizes[n]+sizes[n+1])
# 			b=np.random.randn( sizes[n+1] )  
# 			# w= np.subtract(np.multiply( np.random.rand(sizes[n],sizes[n+1]),2),1)
# 			# b= np.subtract(np.random.rand(sizes[n+1]),0.5)
# 			wt[n]=w
# 			bias[n]=b
# 		wt = np.array(wt)
# 		bias = np.array(bias)
# 	for n in range(num_hidden+1):
# 		if opt =="momentum":
# 			momentum_w.append(np.zeros((sizes[n],sizes[n+1])))
# 			momentum_b.append(np.zeros((sizes[n+1])))
# 		elif opt == "nag":
# 			look_w.append(wt[n])
# 			look_b.append(b[n])
# 			update_w.append(np.zeros((sizes[n],sizes[n+1])))
# 			update_b.append(np.zeros((sizes[n+1])))
# 		elif opt == "adam":
# 			adam_w_m.append(np.zeros((sizes[n],sizes[n+1])))
# 			adam_w_v.append(np.zeros((sizes[n],sizes[n+1])))
# 			adam_b_m.append(np.zeros((sizes[n+1])))
# 			adam_b_v.append(np.zeros((sizes[n+1])))
# 	adam_b_m = np.array(adam_b_m)
# 	adam_b_v = np.array(adam_b_v)
# 	adam_w_v = np.array(adam_w_v)
# 	adam_w_m = np.array(adam_w_m)
# 	if same_model:
# 		if opt == "adam" and os.path.exists(os.path.join("Model4","adam_w_v.npy")):
# 			adam_w_v = np.load(os.path.join("Model4","adam_w_v.npy"))
# 			adam_w_m = np.load(os.path.join("Model4","adam_w_m.npy"))
# 			adam_b_v = np.load(os.path.join("Model4","adam_b_v.npy"))
# 			adam_b_m = np.load(os.path.join("Model4","adam_b_m.npy"))
def initwts():

	global sizes,wt,bias
	global momentum_w,momentum_b
	global look_w,look_b
	global adam_w_m,adam_w_v,adam_b_m,adam_b_v,adam_bp2,adam_bp1
	
	sizes=np.asarray(sizes)
	sizes=np.insert(sizes,0,nof_PCA)
	sizes=np.append(sizes,10).astype(int)

	adam_bp1=0.9
	adam_bp2=0.999
	wt= [None]*(num_hidden+1)
	bias = [None]*(num_hidden+1)
	for n in range(num_hidden+1):
		w=np.random.randn(sizes[n],sizes[n+1])/np.sqrt(sizes[n]+sizes[n+1])
		b=np.random.randn( sizes[n+1] )  
		# w= np.subtract(np.multiply( np.random.rand(sizes[n],sizes[n+1]),2),1)
		# b= np.subtract(np.random.rand(sizes[n+1]),0.5)
		wt[n]=w
		bias[n]=b
	for n in range(num_hidden+1):
		if opt =="momentum":
			momentum_w.append(np.zeros((sizes[n],sizes[n+1])))
			momentum_b.append(np.zeros((sizes[n+1])))
		elif opt == "nag":
			look_w.append(wt[n])
			look_b.append(b[n])
			update_w.append(np.zeros((sizes[n],sizes[n+1])))
			update_b.append(np.zeros((sizes[n+1])))
		elif opt == "adam":
			adam_w_m.append(np.zeros((sizes[n],sizes[n+1])))
			adam_w_v.append(np.zeros((sizes[n],sizes[n+1])))
			adam_b_m.append(np.zeros((sizes[n+1])))
			adam_b_v.append(np.zeros((sizes[n+1])))

# def outputError(y,oneH):
	# pass
def optimizer(k,Dwk,Dbk):
	global wt,bias,momentum_w,momentum_b
	Dwk = np.add(Dwk,np.multiply(wt[k],0.0001))
	Dbk = np.add(Dbk,np.multiply(bias[k],0.0001))
	# print("optimizer",opt,k,Dwk[1,1:4])
	if opt == "gd":
		wt[k]=np.subtract(wt[k],np.multiply(lr,Dwk))
		bias[k]=np.subtract(bias[k],np.multiply(lr,Dbk))
	elif opt == "momentum":
		momentum_w[k]=np.multiply(momentum_w[k],momentum)
		momentum_w[k]=np.add(momentum_w[k],np.multiply(lr,Dwk))
		wt[k]=np.subtract(wt[k],momentum_w[k])        
		momentum_b[k]=np.multiply(momentum_b[k],momentum)
		momentum_b[k]=np.add(momentum_b[k], np.multiply(lr,Dbk))
		bias[k]=np.subtract(bias[k], momentum_b[k])
	elif opt == "nag":
		update_w[k]= np.add( np.multiply(momentum,update_w[k]),np.multiply(lr,Dwk) )
		look_w[k]= np.subtract(look_w[k],update_w[k])
		update_b[k]= np.add( np.multiply(momentum,update_b[k]),np.multiply(lr,Dbk) )
		look_b[k]= np.subtract(look_b[k],update_b[k])
	elif opt == "adam":
		adam_w_m[k] = np.add(np.multiply(adam_b1,adam_w_m[k]),np.multiply(1-adam_b1,Dwk))
		adam_b_m[k] = np.add(np.multiply(adam_b1,adam_b_m[k]),np.multiply(1-adam_b1,Dbk))
		
		adam_w_v[k] = np.add(np.multiply(adam_b2,adam_w_v[k]),np.multiply(1-adam_b2,np.multiply(Dwk,Dwk)))
		adam_b_v[k] = np.add(np.multiply(adam_b2,adam_b_v[k]),np.multiply(1-adam_b2,np.multiply(Dbk,Dbk)))
		
		wt[k] = np.subtract(  wt[k],  np.multiply( np.multiply(lr, np.reciprocal( np.sqrt( np.add( np.divide(adam_w_v[k],1-adam_bp2) ,adam_epsilon) )) ), np.divide(adam_w_m[k],1-adam_bp1) ) )
		bias[k] = np.subtract(bias[k],np.multiply( np.multiply(lr, np.reciprocal( np.sqrt( np.add( np.divide(adam_b_v[k],1-adam_bp2) ,adam_epsilon) )) ), np.divide(adam_b_m[k],1-adam_bp1) ) )
		
# implementing functions to do different tasks. This is the main function block
#def vanilla_grad_desc(num_hidden,sizes):
def grad_desc():
	global freqClass,gloss,adam_bp1,adam_bp2
	x=mini[0:,0:nof_PCA]
	x=np.divide(np.subtract(x.astype(float),127),128)
	y=mini[0:,nof_PCA]

	hs=[]    
	h=x
	hs.append(h)
	
	if opt == "nag":
		for k in range(num_hidden+1):
			# print(type(look_w),update_w[k])
			wt[k] = np.subtract(look_w[k],np.multiply(momentum,update_w[k]) )
			bias[k] = np.subtract(look_b[k],np.multiply(momentum,update_b[k]) )

	#forward Propagation
	for n in range(num_hidden):
		a=np.add(np.matmul(h,wt[n]),bias[n])
		h=fval(a,activation)
		hs.append(h)

	a=np.add(np.matmul(h,wt[num_hidden]),bias[num_hidden]) 
	yhat = np.exp(a) / np.sum (np.exp(a),axis=1,keepdims=True)

	oneH = np.zeros((x.shape[0],10))
	oneH[range(x.shape[0]),y.astype(int)]=1

	if loss == "ce":
		closs = np.sum(-np.log(yhat[range(x.shape[0]),y.astype(int)]))
		Dak = yhat -  oneH
	elif loss == "sq":
		a1 = yhat[:,None,:]
		a2 = yhat[:,:,None]
		a3 = np.eye(yhat.shape[1]) * yhat[:,None,:]
		Dak = np.squeeze(np.matmul(np.subtract(a3, np.matmul(a2,a1)) ,np.subtract(yhat,oneH)[:,:,None]))
		closs = np.sum( np.multiply( np.subtract(yhat,oneH),np.subtract(yhat,oneH) ) )/2

	gloss +=closs

	yt = np.zeros((x.shape[0],10))
	yt[range(x.shape[0]),np.argmax(yhat,axis=1)]=1

	freqClass += np.sum(yt,axis=0)

	nof= np.sum(np.multiply(yt,oneH))
	
	for k in range(num_hidden,-1,-1):
		Dwk = np.matmul(hs[k].T,Dak)
		Dwk = Dwk/x.shape[0]
		Dbk = Dak
		Dbk = np.mean(Dbk,axis=0)
		optimizer(k,Dwk,Dbk)
		Dhk = np.matmul(Dak , wt[k].T)

		Dak = np.multiply(Dhk,  fgrad(hs[k],activation) )       
		
	if opt == "adam":
		adam_bp1=adam_bp1*adam_b1
		adam_bp2=adam_bp2*adam_b2
	return (closs,1.0-(nof*1.0)/x.shape[0],nof)

def validation(data,classlbl=False):
	global expt_dir,ploss,lr,anneal,iii,nofp,nofpp
	global adam_b_m, adam_b_v, adam_w_m, adam_w_v ,wt, bias,  adam_bp1 , adam_bp2
	global adam_b_mc,adam_w_mc,adam_w_vc,adam_b_vc,wtc,biasc, adam_bp1c,adam_bp2c
	x=data[0:,0:nof_PCA]
	x=np.divide(np.subtract(x.astype(float),127),128)

	if not classlbl:
		y=data[0:,nof_PCA]

	h=x
	for n in range(num_hidden):
		a=np.add(np.matmul(h,wt[n]),bias[n])
		h=fval(a,activation)

	a=np.add(np.matmul(h,wt[num_hidden]),bias[num_hidden]) 
	yhat = np.exp(a) / np.sum (np.exp(a),axis=1,keepdims=True)



	yt = np.zeros((x.shape[0],10))
	yt[range(x.shape[0]),np.argmax(yhat,axis=1)]=1

	freqClass = np.sum(yt,axis=0)
	if classlbl:
		print("test",data.shape,freqClass)
		return np.argmax(yhat,axis=1)
		
	oneH = np.zeros((x.shape[0],10))
	oneH[range(x.shape[0]),y.astype(int)]=1

	nofp= np.sum(np.multiply(yt,oneH))

	if loss == "ce":
		closs = np.sum(-np.log(yhat[range(x.shape[0]),y.astype(int)]))
	elif loss == "sq":
		closs = np.sum( np.multiply( np.subtract(yhat,oneH),np.subtract(yhat,oneH) ) )/2



	if anneal:
		print("anneal",ploss,closs,nofp,nofpp)
		if ploss < closs and nofp < nofpp:
			wt = np.array(wtc)
			bias = np.array(biasc)
			if opt == "adam":
				adam_bp2 =adam_bp2c
				adam_bp1 =adam_bp1c
				adam_b_m = np.array(adam_b_mc)
				adam_b_v = np.array(adam_b_vc)
				adam_w_m = np.array(adam_w_mc)
				adam_w_v = np.array(adam_w_vc)

			print("anneal",ploss,closs)
			lr = lr*0.9
			iii-=1

		wtc = np.array(wt)
		biasc = np.array(bias)
		if opt == "adam":
			adam_bp2c =adam_bp2
			adam_bp1c =adam_bp1
			adam_b_mc = np.array(adam_b_m)
			adam_b_vc = np.array(adam_b_v)
			adam_w_mc = np.array(adam_w_m)
			adam_w_vc = np.array(adam_w_v)


	nofpp = nofp
	ploss = closs

	return (closs,1.0 - (nofp*1.0)/x.shape[0])
	# print("success:" ,float(nofp)/x.shape[0],closs,"\n",freqClass)


def main():
	global lr,momentum,num_hidden,sizes,activation,loss,opt,batch_size,epoch,anneal,save_dir,expt_dir,train_path,test_path,valid_path
	global train,test,valid,wt,bias,adam_bp1,adam_bp2,steps_per_batch,nofp
	global iii,jj,mini,file,freqClass,gloss,step,train,test,valid
	print("parsing...")
	parser = agp.ArgumentParser()
	parser.add_argument("--lr", type=float, help="the learning rate", default=0.001)
	parser.add_argument("--momentum", type=float, help="the momentum in lr", default=0.9)
	parser.add_argument("--num_hidden", type=int, help="# of Hidden Layers", default=2)
	parser.add_argument("--sizes", type=csv_list, help="# of Nodes per H_Layer", default= [90,80])
	parser.add_argument("--activation", type=str, help="activation function", default= "sigmoid", choices=["sigmoid","tanh","relu"])
	parser.add_argument("--loss", type=str, help="loss function", default= "ce", choices=["sq","ce"])
	parser.add_argument("--opt", type=str, help="optimizer", default= "adam", choices=["gd","momentum","nag","adam"])
	parser.add_argument("--batch_size", type=int, help="batch size per step", default= 20)
	parser.add_argument("--epoch", type=int, help="# of Epochs", default= 100)
	parser.add_argument("--anneal", type=annealf, help="anneal", default= False,choices=[True,False])
	parser.add_argument("--save_dir", type=str, help="Save dir location", default= "Results")
	parser.add_argument("--expt_dir", type=str, help="expt_dir location", default= os.path.join("Results","Log"))
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
	
	valid=pd.read_csv(valid_path)
	print("finished reading images...")

	# train=train.values

	train=train.values
	valid=valid.values
	x=train[:,1:785]
	y=train[:,785]
	
	
	if os.path.exists(os.path.join("Model4","PCA.sav")):
		print("PCA exists")
		pcamod = joblib.load(os.path.join("Model4","PCA.sav"))
	else:
		pca=PCA(n_components=nof_PCA)
		pcamod=pca.fit(x) #pickle this to use it on test data
		print("Dumping  PCA")
		joblib.dump(pcamod, os.path.join("Model4","PCA.sav"))
	x=pcamod.transform(x)
	yn = y.reshape(x.shape[0],1)
	train=  np.hstack((x,yn))

	x_val=valid[:,1:785]
	y_val =valid[:,785]
	x_val =pcamod.transform(x_val)
	yn = y_val.reshape(y_val.shape[0],1)
	valid=  np.hstack((x_val,yn))

	# print(" Correctly classified samples ratio is %d \n"%round((nof/55000),2))

	steps_per_batch=int(train.shape[0]/batch_size)

#=====================
	for sizes in [[50],[100],[200],[300],[50,50],[100,100],[200,200],[300,300]]:
		tloss=[]
		vloss=[]
		num_hidden = len(sizes) 
		initwts()
		nofp=0#comes from validation
		lloss=0
		step=0 # update step for printing loss

		cum_loss_step=0
		cum_error_step=0
		for iii in range(epoch):
			np.random.shuffle(train)
			freqClass = np.zeros(10)
			gloss=0
			nof_ep=0
			cum_loss_epoch=0
			for jj in range(steps_per_batch):
				step=step+1
				if(step%100==0):
					logfile(expt_dir,"_"+ activation+"_" +loss+"("+",".join([str(e) for e in sizes[:-1]])+")_"+"train.txt" ,cum_loss_step,cum_error_step/100)
					(lossv,nof) = validation(valid)
					cum_error_step=0
					cum_loss_step=0
					logfile(expt_dir,"_"+ activation+"_" +loss+"("+",".join([str(e) for e in sizes[:-1]])+")_"+"valid.txt" ,lossv,nof)

				mini = train[jj*batch_size:(jj+1)*batch_size,:]
				(lloss,nof,pred)=grad_desc()
				cum_error_step+=nof
				cum_loss_step+=lloss
				nof_ep+=pred
				cum_loss_epoch+=lloss
			print("\n: ",iii,nof_ep,freqClass.astype(int),gloss,(nofp*100.0)/(valid.shape[0])  )
			tloss.append(cum_loss_epoch)	
			vloss.append(lossv)	

		np.save(os.path.join(expt_dir,"_"+ activation+"_" +loss+"("+",".join([str(e) for e in sizes[:-1]])+")_"+"tloss.npy"),np.array(tloss))
		np.save(os.path.join(expt_dir,"_"+ activation+"_" +loss+"("+",".join([str(e) for e in sizes[:-1]])+")_"+"vloss.npy"),np.array(vloss))

		print("===============\n=================\n=================\n==================")
#====================

		# if (iii %50) == 20:
		# 	with open(os.path.join("Model4","model.pkl"), 'wb') as f:
		# 		model = {}
		# 		model["sizes"]=sizes
		# 		model["num_hidden"]=num_hidden
		# 		if opt =="adam":
		# 			model["adam_bp1"]=adam_bp1
		# 			model["adam_bp2"]=adam_bp2
		# 			np.save(os.path.join("Model4","adam_w_v.npy"),np.array(adam_w_v))
		# 			np.save(os.path.join("Model4","adam_b_v.npy"),np.array(adam_b_v))
		# 			np.save(os.path.join("Model4","adam_w_m.npy"),np.array(adam_w_m))
		# 			np.save(os.path.join("Model4","adam_b_m.npy"),np.array(adam_b_m))

		# 		pickle.dump(model, f)
		# 		np.save(os.path.join("Model4","wt.npy"),np.array(wt))
		# 		np.save(os.path.join("Model4","bias.npy"),np.array(bias))
		

		# if gloss<40 and nof_ep>=train.shape[0]-2 and iii > 100:
			# break
	# for i in range(10):
		# yc=vanilla_grad_desc(num_hidden,sizes)
	
	#test set prediction
	# test=pd.read_csv(test_path)
	# test=test.values
	# sno=test[:,0]
	# x=test[:,1:785]
	# x=pcamod.transform(x)
	# ypred=validation(x,classlbl=True)

	# fn="test_submission_"+activation+"_"+"_".join([str(e) for e in sizes[:-1]])+"_"+str(int((nofp*10000)/(valid.shape[0])) )+".csv"
	# f_location  = os.path.join(expt_dir,fn)
	# fpred=open(f_location,'a+')
	# fpred.write("id,label\n")
	# for r in range(x.shape[0]):
	# 	fpred.write("%d,%d\n"%(sno[r],ypred[r]) )
	# # print("  ",ypred[2000:2500],sno[])
	# fpred.close()

	print(sizes,"sizes")

def logfile(expt_dir,log_type,lloss,err): #log_type: log_train.txt/log_validation.txt depending upon the data used
	f_location=os.path.join(expt_dir,log_type)
	f=open(f_location , 'a+')
	f.write(" Epoch : %d , Step : %d , Loss : %f , Error: %f , lr :%f\n" %(iii,step,lloss,err,lr))
	f.close()   
	
if __name__=="__main__":
	main()
	







