from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def loaddata(fname):
    data = np.loadtxt(fname,skiprows=1, delimiter=',')
    y = data[:,0]
    X = data[:,1:]
    nm = np.sqrt(np.sum(X * X, axis=1))
    X = X / nm[:,None]
    return y, X

def svm_sgd(data, labels, C, eta, passes):
    nd, nf = np.shape(data)
    
    T = nd
    w0 = 0
    w = np.zeros(nf)
    lst=[]
    for i in range(passes):
     	numerr = 0
     	loss=0.0
     	total_loss=0.0
     	for t in range(T):
        	dw = -2*w/(nd*C)
        	loss=(1-labels[t]*(np.dot(w,data[t])+w0))
        	if loss> 0:
            		dw += labels[t]*data[t]
            		w0 += eta*labels[t]
            		numerr += 1
            		total_loss+=loss
        	w += eta * dw
     	#print "i", i
     	#print "Total_loss", total_loss
     	lst.append(np.linalg.norm(w,2)/(nd*C)+total_loss/nd)
    	#print lst
    print "Errors ", numerr
    print "L2 Norm", np.linalg.norm(w,2)
    return (w0, w, lst,numerr)

def predict(W,W0,data,labels):
	#labels are just being passed for checking the errors
	errs=0
	nd,nf=np.shape(data)
	T=nd
	for t in range(T):
		pred=np.dot(W,data[t])+W0
		#print 'pred is %d and label is %d' %(pred, labels[t])
		#return
		if(pred>0 and labels[t]<0):
			errs+=1
	return errs

def linear_svm(trainX,trainY,testX,testY,passes):
    
    C=1
    eta = 0.03
    ts=[i for i in range(passes)]
    avglst=[]
    w0, w, avgls,errs = svm_sgd(trainX, trainY, C, eta,passes=passes)
    
    #avglst.append(avgls)
    plt.plot(ts,avgls)
    plt.xlabel('Passes')
    plt.ylabel('Average Loss')
    plt.title('Linear SVM With Stochastic Gradient Descent')    
    plt.grid(True)
    plt.show()
    print 'Error on Validation Data after complete passes are ', errs

    errtest=predict(w,w0,testX,testY)
    print 'err rate on testing is ', errtest
    
    
trainY, trainX = loaddata("validation.csv")
print 'TrainX shape', trainX.shape
testY, testX = loaddata("test.csv")
linear_svm(trainX,trainY,testX,testY,passes=50)
