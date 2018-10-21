import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn import svm
import statistics as stat


def makedata(num):
    X = []
    Y = []
    for i in range(num):
        a = random.uniform(-2,2)
        b = random.uniform(-2,2)
        X.append([a, b])
    X = np.array(X)
    
    a = random.uniform(1,2)
    b = random.uniform(0,1)
    c = random.uniform(0,1)
    d = random.uniform(2,4)
    e = random.uniform(0,1)
    for i in range(len(X)):
        if(a*(X[i][0])**5+b*(X[i][0])**4+c*(X[i][0])**3+d*(X[i][0])**2+e*(X[i][0]))- X[i][1] >0:
            Y.append(1)
        else:
            Y.append(0)              
    Y = np.array(Y)
    
    for i in range(len(X)):
        X[i] = X[i]+np.random.normal(scale=0.2 , size=2)
    
    return X,Y,a,b,c,d,e

def make_graph(num):
    X,Y,o,p,q,r,s = makedata(100)
    print(o,"x^5",p,"x^4",q,"x^3",r,"x^2",s,"x")
    plt.scatter(X[:len(X)//2, 0], X[:len(X)//2, 1], marker='o', c=Y[:len(Y)//2] ,s=25, edgecolor='k')
    plt.scatter(X[len(X)//2:,0],X[len(X)//2:,1],marker='x', c=Y[len(Y)//2:] ,s=25, edgecolor='r')
    
    clf = svm.NuSVC(nu=0.1, degree=9,kernel='poly')
    clf.fit(X[:len(X)//2], Y[:len(Y)//2])
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()   
    
    xx = np.linspace(xlim[0], xlim[1], 20)
    yy = np.linspace(ylim[0], ylim[1], 20)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    #ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5,
    #           linestyles=['-'])
    xxx = np.linspace(-5,5,100)
    plt.ylim([-4,4])
    plt.plot(xxx,o*(xxx**5)+p*(xxx**4)+p*(xxx**3)+r*(xxx**2)+s*(xxx),'g', linewidth =2) 
    test = clf.score(X[len(X)//2:],Y[len(Y)//2:])
    train = clf.score(X[:len(X)//2], Y[:len(Y)//2])
    print("training error: ",train)
    print("test error: ",test)
    plt.show()

def get_error(num,nv,deg):
    X1, Y1, o, p ,q,r ,s = makedata(num)
    
    #gaussian_noise_train = np.random.normal(scale=noise , size=2)
    #gaussian_noise_test = np.random.normal(scale=noise , size=2)  
    #X1[:len(X1)//4] = X1[:len(X1)//4] + gaussian_noise_train
    #X1[len(X1)//2:] = X1[len(X1)//2:] + gaussian_noise_test 
    
    clf = svm.NuSVC(nu=nv, degree=deg, kernel='poly')
    clf.fit(X1[:len(X1)//2], Y1[:len(Y1)//2]) 
    test = clf.score(X1[len(X1)//2:],Y1[len(Y1)//2:])
    train = clf.score(X1[:len(X1)//2], Y1[:len(Y1)//2])
    #print(train,test)
    #print(test)
    return train,test

def test_deg():
    N = 50
    p_ave_list = []
    p_sd_list = []
    num_trails = 50
 
    i_array = np.arange(1, 11, 2)
    for i in i_array:
        p_list = []
        print(i)
        count = 0
        while (count < num_trails):
            try:  
                a, b = get_error(N, 0.2, i)
                p_list.append(b)
                count = count + 1
            except ValueError:
                print("error")
                continue               
        p_ave = stat.mean(p_list)
        p_sd = stat.stdev(p_list)
        p_ave_list.append(p_ave)
        p_sd_list.append(p_sd)

    plt.errorbar(i_array, p_ave_list, yerr = p_sd_list, 
                 fmt = 'o', color = 'r') 
    #plt.title("Nu = 0.1")
    plt.xlabel('degree')
    plt.ylabel('Accuracy')
    plt.show() 
        
def cross_get(num,nv,deg):
    X1, Y1, o, p ,q,r ,s = makedata(num)
    X11 = X1[:len(X1)//2]
    Y11 = Y1[:len(Y1)//2]
    
    clf = svm.NuSVC(nu=nv, degree=deg, kernel='poly')
    clf.fit(X11[:len(X11)//2], Y11[:len(Y11)//2]) 
    train = clf.score(X11[:len(X11)//2], Y11[:len(Y11)//2])
    vali = clf.score(X11[len(X11)//2:], Y11[len(Y11)//2:])
    test = clf.score(X1[len(X11)//2:],Y1[len(Y11)//2:])

    #print(train,test)
    return train,vali,test

def deg_cross(num,deg):
    best_train = 0
    best_vali = 0
    best_test = 0
    nu = 0
    i_array = np.arange(0.01,0.4,0.05)
    for i in i_array:
        try:
            train,vali,test = cross_get(num,i, deg)
            if vali > best_vali:
                best_vali = vali
                best_train = train
                best_test = test
                nu = i
        except ValueError:
                print("error")
                continue
           
    #print(nu)
    return best_train,best_vali,best_test

def test_deg_cross():
    N = 100
    p_ave_list = []
    p_sd_list = []
    num_trails = 100
 
    i_array = np.arange(1,11, 2)
    for i in i_array:
        p_list = []
        print(i)
        count = 0
        while (count < num_trails):
            #try:  
                a, c, b = deg_cross(N,i)
                p_list.append(a)
                count = count + 1
            #except ValueError:
                #continue               
        p_ave = stat.mean(p_list)
        p_sd = stat.stdev(p_list)
        p_ave_list.append(p_ave)
        p_sd_list.append(p_sd)

    plt.errorbar(i_array, p_ave_list, yerr = p_sd_list, 
                 fmt = 'o', color = 'r') 
    #plt.title("Nu = 0.1")
    plt.xlabel('degree')
    plt.ylabel('training accuracy')
    plt.show()     
    
if __name__ == '__main__':
    #make_graph(100)
    #get_error(20,0.1,3)
    #test_deg()
    for i in range(10):
        test_deg_cross()
