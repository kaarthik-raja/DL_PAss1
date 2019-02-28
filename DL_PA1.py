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


#set random seed for replicability of results
np.random.seed(1234)
gamma=0.9
eta=0.0001
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
def initwts():
	global sizes,wt,bias
	global momentum_w,momentum_b
	global look_w,look_b
	global adam_w_m,adam_w_v,adam_b_m,adam_b_v,adam_bp2,adam_bp1
	
	sizes=np.asarray(sizes)
	sizes=np.insert(sizes,0,nof_PCA)
	sizes=np.append(sizes,10).astype(int)

	same_model = False
	if os.path.exists(os.path.join("Model4","model.pkl")):
		with open(os.path.join("Model4","model.pkl"), 'rb') as f:
			model=pickle.load(f)
			nsizes  = model["sizes"]
			adam_bp2 = model.get("adam_bp2",adam_b2)
			adam_bp1 = model.get("adam_bp1",adam_b1)
			same_model = True
			if num_hidden == model["num_hidden"]:
				for k in range(num_hidden+1):
					if nsizes[k] != sizes[k]:
						same_model = False
			else:
				same_model = False
			print("Same  Model :",same_model)
	if same_model:
		wt = np.load(os.path.join("Model4","wt.npy"))
		bias = np.load(os.path.join("Model4","bias.npy"))
		print("continuing previous model data....")
	else:
		wt= [None]*(num_hidden+1)
		bias = [None]*(num_hidden+1)
		for n in range(num_hidden+1):
			w=np.random.randn(sizes[n],sizes[n+1])/np.sqrt(sizes[n]+sizes[n+1])
			b=np.random.randn( sizes[n+1] )  
			# w= np.subtract(np.multiply( np.random.rand(sizes[n],sizes[n+1]),2),1)
			# b= np.subtract(np.random.rand(sizes[n+1]),0.5)
			wt[n]=w
			bias[n]=b
		wt = np.array(wt)
		bias = np.array(bias)
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
	adam_b_m = np.array(adam_b_m)
	adam_b_v = np.array(adam_b_v)
	adam_w_v = np.array(adam_w_v)
	adam_w_m = np.array(adam_w_m)
	if same_model:
		if opt == "adam" and os.path.exists(os.path.join("Model4","adam_w_v.npy")):
			adam_w_v = np.load(os.path.join("Model4","adam_w_v.npy"))
			adam_w_m = np.load(os.path.join("Model4","adam_w_m.npy"))
			adam_b_v = np.load(os.path.join("Model4","adam_b_v.npy"))
			adam_b_m = np.load(os.path.join("Model4","adam_b_m.npy"))

def fgrad(hs):
	if activation == "sigmoid":
		return np.multiply(hs,1-hs)
	elif activation == "tanh":
		return np.subtract(1,np.multiply(hs , hs))
	else:
		hs[hs<=0]=0
		hs[hs>0]=1
		return hs

def fval(a):
	if activation == "sigmoid":
		return np.reciprocal(np.add(1,np.exp(np.negative(a) )))
	elif activation == "tanh":
		return np.multiply(np.subtract(np.exp(2*a),1) ,  np.reciprocal( np.add( np.exp(2*a),1) )  )
	else:
		a[a<=0]=0
		return a

# def outputError(y,oneH):
	# pass
def optimizer(k,Dwk,Dbk):
	global wt,bias,momentum_w,momentum_b
	Dwk = np.add(Dwk,np.multiply(wt[k],0.0001))
	# print("optimizer",opt,k,Dwk[1,1:4])
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
		update_w[k]= np.add( np.multiply(gamma,update_w[k]),np.multiply(eta,Dwk) )
		look_w[k]= np.subtract(look_w[k],update_w[k])
		update_b[k]= np.add( np.multiply(gamma,update_b[k]),np.multiply(eta,Dbk) )
		look_b[k]= np.subtract(look_b[k],update_b[k])
	elif opt == "adam":
		adam_w_m[k] = np.add(np.multiply(adam_b1,adam_w_m[k]),np.multiply(1-adam_b1,Dwk))
		adam_b_m[k] = np.add(np.multiply(adam_b1,adam_b_m[k]),np.multiply(1-adam_b1,Dbk))
		
		adam_w_v[k] = np.add(np.multiply(adam_b2,adam_w_v[k]),np.multiply(1-adam_b2,np.multiply(Dwk,Dwk)))
		adam_b_v[k] = np.add(np.multiply(adam_b2,adam_b_v[k]),np.multiply(1-adam_b2,np.multiply(Dbk,Dbk)))
		
		wt[k] = np.subtract(  wt[k],  np.multiply( np.multiply(eta, np.reciprocal( np.sqrt( np.add( np.divide(adam_w_v[k],1-adam_bp2) ,adam_epsilon) )) ), np.divide(adam_w_m[k],1-adam_bp1) ) )
		bias[k] = np.subtract(bias[k],np.multiply( np.multiply(eta, np.reciprocal( np.sqrt( np.add( np.divide(adam_b_v[k],1-adam_bp2) ,adam_epsilon) )) ), np.divide(adam_b_m[k],1-adam_bp1) ) )
		
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
			wt[k] = np.subtract(look_w[k],np.multiply(gamma,update_w[k]) )
			bias[k] = np.subtract(look_b[k],np.multiply(gamma,update_b[k]) )

	#forward Propagation
	for n in range(num_hidden):
			a=np.add(np.matmul(h,wt[n]),bias[n])
			h=fval(a)
			hs.append(h)

	a=np.add(np.matmul(h,wt[num_hidden]),bias[num_hidden]) 
	yhat = np.exp(a) / np.sum (np.exp(a),axis=1,keepdims=True)

	loss = np.sum(-np.log(yhat[range(x.shape[0]),y.astype(int)]))
	gloss +=loss
	oneH = np.zeros((x.shape[0],10))
	oneH[range(x.shape[0]),y.astype(int)]=1

	yt = np.zeros((x.shape[0],10))
	yt[range(x.shape[0]),np.argmax(yhat,axis=1)]=1

	freqClass += np.sum(yt,axis=0)

	nof= np.sum(np.multiply(yt,oneH))
	nofc[iii]+=nof    
	

	#backward Propagation
	Dak = yhat -  oneH
	
	for k in range(num_hidden,-1,-1):
		Dwk = np.matmul(hs[k].T,Dak)
		Dwk = Dwk/x.shape[0]
		Dbk = Dak
		Dbk = np.mean(Dbk,axis=0)
		optimizer(k,Dwk,Dbk)
		Dhk = np.matmul(Dak , wt[k].T)

		Dak = np.multiply(Dhk,  fgrad(hs[k]) )       
		
	if opt == "adam":
		adam_bp1=adam_bp1*adam_b1
		adam_bp2=adam_bp2*adam_b2
	return loss

def validation(data,classlbl=False):
	global adam_bp1,adam_bp2,expt_dir,ploss,eta,anneal,iii,nofp,nofpp
	global adam_b_mc,adam_w_mc,adam_w_vc,adam_b_vc,wtc,biasc,adam_bp1c,adam_bp2c
	x=data[0:,0:nof_PCA]
	x=np.divide(np.subtract(x.astype(float),127),128)

	if not classlbl:
		y=data[0:,nof_PCA]

	h=x
	for n in range(num_hidden):
			a=np.add(np.matmul(h,wt[n]),bias[n])
			h=fval(a)

	a=np.add(np.matmul(h,wt[num_hidden]),bias[num_hidden]) 
	yhat = np.exp(a) / np.sum (np.exp(a),axis=1,keepdims=True)



	yt = np.zeros((x.shape[0],10))
	yt[range(x.shape[0]),np.argmax(yhat,axis=1)]=1

	freqClass = np.sum(yt,axis=0)
	if classlbl:
		print("test",data.shape,freqClass)
		return np.argmax(yhat,axis=1)

	loss = np.sum(-np.log(yhat[range(x.shape[0]),y.astype(int)]))

	if anneal:
		if ploss < loss and nofp < nofpp:


			wt = np.array(wtc)
			bias = np.array(biasc)
			if opt == "adam":
				adam_b_m = np.array(adam_b_mc)
				adam_b_v = np.array(adam_b_vc)
				adam_w_m = np.array(adam_w_mc)
				adam_w_v = np.array(adam_w_vc)

			print("anneal",ploss,loss)
			eta = eta*0.9
			iii-=10

		wtc = np.array(wt)
		biasc = np.array(bias)
		if opt == "adam":
			adam_b_mc = np.array(adam_b_m)
			adam_b_vc = np.array(adam_b_v)
			adam_w_mc = np.array(adam_w_m)
			adam_w_vc = np.array(adam_w_v)

	oneH = np.zeros((x.shape[0],10))
	oneH[range(x.shape[0]),y.astype(int)]=1

	nofp= np.sum(np.multiply(yt,oneH))
	logfile(expt_dir,"log_validation.txt")
	nofpp = nofp
	ploss = loss
	print("success:" ,float(nofp)/x.shape[0],"\n",freqClass)

def csv_list(string):
   return [ int(i) for i in string.split(',')]

def annealf(string):
	if string in ["true","True","T","t","1" ] :
		return True
	elif string in ["False","false","F","f","0"]:
		return False

def logfile(expt_dir,log_type): #log_type: log_train.txt/log_validation.txt depending upon the data used
	f_location='%s%s' %(expt_dir,log_type)
	f=open(f_location , 'a+')
	f.write(" Epoch : %d , Step : %d , Loss : %f , Error: %f , lr :%f" %(iii,step,gloss,(55000-nofc[iii])/55000,eta))
	f.close()    

def testprediction(expt_dir):
	pass

def main():
	global lr,momentum,num_hidden,sizes,activation,loss,opt,batch_size,epoch,anneal,save_dir,expt_dir,train_path,test_path,valid_path
	global train,test,valid,wt,bias,adam_bp1,adam_bp2,steps_per_batch,nofp
	global iii,jj,mini,nofc,file,freqClass,gloss,step,train,test,valid
	print("parsing...")
	parser = agp.ArgumentParser()
	parser.add_argument("--lr", type=float, help="the learning rate", default=0.01)
	parser.add_argument("--momentum", type=float, help="the momentum in lr", default=0.5)
	parser.add_argument("--num_hidden", type=int, help="# of Hidden Layers", default=2)
	parser.add_argument("--sizes", type=csv_list, help="# of Nodes per H_Layer", default= [90,80])
	parser.add_argument("--activation", type=str, help="activation function", default= "relu", choices=["sigmoid","tanh","relu"])
	parser.add_argument("--loss", type=str, help="loss function", default= "ce", choices=["sq","ce"])
	parser.add_argument("--opt", type=str, help="optimizer", default= "gd", choices=["gd","momentum","nag","adam"])
	parser.add_argument("--batch_size", type=int, help="batch size per step", default= 100)
	parser.add_argument("--epoch", type=int, help="# of Epochs", default= 1000)
	parser.add_argument("--anneal", type=annealf, help="anneal", default= False,choices=[True,False])
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

	file = open("train.txt","w")
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
	initwts()
	nofc=np.zeros(epoch)
	nofp=0
	print("dets","test_submission_"+activation+"_"+"_".join([str(e) for e in sizes])+"_"+str(int((nofp*10000)/(valid.shape[0])) )+".csv")
	step=0 # update step for printing loss
	for iii in range(epoch):
		np.random.shuffle(train)
		freqClass = np.zeros(10)
		gloss=0
		for jj in range(steps_per_batch):
			step=step+1
			if(step%100==0):
				logfile(expt_dir,"log_train.txt")
			mini = train[jj*batch_size:(jj+1)*batch_size,:]
			lloss=grad_desc()


		if (iii%10)==0:
			validation(valid)
		if (iii %100) == 20:
			with open(os.path.join("Model4","model.pkl"), 'wb') as f:
				model = {}
				model["sizes"]=sizes
				model["num_hidden"]=num_hidden
				if opt =="adam":
					model["adam_bp1"]=adam_bp1
					model["adam_bp2"]=adam_bp2
					np.save(os.path.join("Model4","adam_w_v.npy"),np.array(adam_w_v))
					np.save(os.path.join("Model4","adam_b_v.npy"),np.array(adam_b_v))
					np.save(os.path.join("Model4","adam_w_m.npy"),np.array(adam_w_m))
					np.save(os.path.join("Model4","adam_b_m.npy"),np.array(adam_b_m))

				pickle.dump(model, f)
				np.save(os.path.join("Model4","wt.npy"),np.array(wt))
				np.save(os.path.join("Model4","bias.npy"),np.array(bias))

		print("\n: ",iii,nofc[iii],freqClass.astype(int),gloss)
		if gloss<40 and nofc[iii]>=train.shape[0]-2 and iii > 100:
			break
	# for i in range(10):
		# yc=vanilla_grad_desc(num_hidden,sizes)
	
	file.close()
	# plt.pyplot(iii,nofc)

	#test set prediction
	test=pd.read_csv(test_path)
	test=test.values
	sno=test[:,0]
	x=test[:,1:785]
	x=pcamod.transform(x)
	ypred=validation(x,classlbl=True)

	fn="test_submission_"+activation+"_"+"_".join([str(e) for e in sizes[:-1]])+"_"+str(int((nofp*10000)/(valid.shape[0])) )+".csv"
	f_location  = os.path.join(expt_dir,fn)
	fpred=open(f_location,'a+')
	fpred.write("id,label\n")
	for r in range(x.shape[0]):
		fpred.write("%d,%d\n"%(sno[r],ypred[r]) )
	# print("  ",ypred[2000:2500],sno[])
	fpred.close()

	print(sizes,"sizes")

	
if __name__=="__main__":
	main()
	







