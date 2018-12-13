# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 21:17:32 2018

@author: Adi
"""
import numpy as np
import scipy.special

ip=[[0,0],[0,1],[1,0],[1,1]]
t=[0,0,0,1]
wt_i2h=[[1,1,1],[1,1,1]]
wt_h2o=[1,1]
hid=3
h=np.zeros(hid)
alpha=0.1
bias = 1
iterate=5000
def activation(x):
    return scipy.special.expit(x)

for n in range(iterate):
    print("Iteration{0}".format(n))
        
    for k in ip:
        c=0
        #for each input k
        #for each hidden unit hi find hin=w_i2h[i][j]*inp[k][j]
        for i in range(len(h)):
            hin=0
            for j in range(len(k)):#try with len(i2h)
                #print("k{0}*w_i2h{0}{1}".format(j+1,i+1))
                #print(hin)
                hin+=k[j]*wt_i2h[j][i]
                #print("Hin",i+1, "=", hin)
            hin=hin+bias
            h[i]=activation(hin)
            #print("h",i+1,"=",h[i])
            
        #print("hidden val",h,"Weights I2H" ,wt_i2h,"Weights h2o", wt_h2o)
        
           
        yin=0
        for j in range(len(wt_h2o)):
            yin+=wt_h2o[j]*h[j]
            yin+=bias
        #print("yin",yin)
        y=activation(yin)
        #print("y",y)
        delta=alpha*(t[c]-y)
        #print("Delta",delta)
        for i in range(len(wt_i2h)):
            l=len(wt_i2h[0])
            for j in range(l):
                #print("wi2h=",wt_i2h[i][j],"and input is ",k[i])
                wt_i2h[i][j]=wt_i2h[i][j]+delta*k[i]
                print(wt_i2h)
        
        for i in range(len(wt_h2o)):
            #print("wh2o=",wt_h2o[i],"and hidden is ",h[i])
            wt_h2o[i]=wt_h2o[i]+delta*h[i]
            #print(wt_h2o)
        c=c+1
    
        print('\n',"Weights are: \n")
        print("Wi2h=",wt_i2h,"\nwh2o=",wt_h2o)  
        
        print("output y is =", y)
        
        print("-----------------------End-------------------------")
    alpha=alpha/2
        
    