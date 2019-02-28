import numpy as np
from global_m import *

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
