import numpy as np

def fgrad(hs,activation):
	if activation == "sigmoid":
		return np.multiply(hs,1-hs)
	elif activation == "tanh":
		return np.subtract(1,np.multiply(hs , hs))
	else:
		hs[hs<=0]=0
		hs[hs>0]=1
		return hs

def fval(a,activation):
	if activation == "sigmoid":
		return np.reciprocal(np.add(1,np.exp(np.negative(a) )))
	elif activation == "tanh":
		return np.multiply(np.subtract(np.exp(2*a),1) ,  np.reciprocal( np.add( np.exp(2*a),1) )  )
	else:
		a[a<=0]=0
		return a
def csv_list(string):
   return [ int(i) for i in string.split(',')]

def annealf(string):
	if string in ["true","True","T","t","1" ] :
		return True
	elif string in ["False","false","F","f","0"]:
		return False

 

def testprediction(expt_dir):
	pass
