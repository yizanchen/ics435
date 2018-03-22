import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn import svm
import statistics as stat

def data(num):
    date = []
    for i in range(num):
        a = random.uniform(-5,5)
        b = random.uniform(-5,5)
        date.append([a, b])
        
    datas = np.array(date)
    return datas
    
def polys(InputN):
    a = random.uniform(1,4)
    b = random.uniform(1,4)
    c = random.uniform(1,4)
    X1 = data(InputN)
    Y1 = []
    for i in range(len(X1)):
        if(a*(X1[i][0])**3+b*(X1[i][0])**2+c*(X1[i][0]))- X1[i][1] >0:
            Y1.append(1)
        else:
            Y1.append(0)
        
    Y1 = np.array(Y1)
    return X1, Y1, a, b, c
    
    
def makenu():
    X1, Y1, o, p, q = polys(1000)
    #print(str(o) + ' ' + str(p) + ' ' + str(q))
    
    gaussian_noise_train = np.random.normal(scale=0.5 , size=2)
    #gaussian_noise_test = np.random.normal(scale=0.3 , size=2)
    print(gaussian_noise_train)
   # print(gaussian_noise_test)
    #print(X1)
    X1[:len(X1)//4] = X1[:len(X1)//4] + gaussian_noise_train
  #  X1[len(X1)//2:] = X1[len(X1)//2:] + gaussian_noise_test
    #print(X1)
    #print(X1)
    #print(Y1)
    plt.scatter(X1[:len(X1)//2, 0], X1[:len(X1)//2, 1], marker='o', c=Y1[:len(Y1)//2] ,s=25, edgecolor='k')
    plt.scatter(X1[len(X1)//2:,0],X1[len(X1)//2:,1],marker='x', c=Y1[len(Y1)//2:] ,s=25, edgecolor='r')

    clf = svm.NuSVC(nu=0.1, degree=3,kernel='rbf')
    clf.fit(X1[:len(X1)//2], Y1[:len(Y1)//2])
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xlim
    ylim
# create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 20)
    yy = np.linspace(ylim[0], ylim[1], 20)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    Z
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5,
               linestyles=['-'])
    # plot support vectors
    #ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
    #         linewidth=1, facecolors='r')  
    xxx = np.linspace(-5,5,100)
    plt.ylim([-5,5])

    plt.plot(xxx,o*(xxx**3)+p*(xxx**2)+q*(xxx),'g', linewidth =2)
      
    b = clf.score(X1[len(X1)//2:],Y1[len(Y1)//2:])
    a = clf.score(X1[:len(X1)//2], Y1[:len(Y1)//2])
    print("training error: ",a)
    print("test error: ",b)

    plt.show()
 
def makenu2(inputs,noise,nuvalue):
    X1, Y1, o, p ,q = polys(inputs)
    
    gaussian_noise_train = np.random.normal(scale=noise , size=2)
    #gaussian_noise_test = np.random.normal(scale=noise , size=2)
    
    X1[:len(X1)//4] = X1[:len(X1)//4] + gaussian_noise_train
    #X1[len(X1)//2:] = X1[len(X1)//2:] + gaussian_noise_test 
    clf = svm.NuSVC(nu=nuvalue, degree=6,kernel='rbf')
    clf.fit(X1[:len(X1)//2], Y1[:len(Y1)//2]) 
    b = clf.score(X1[len(X1)//2:],Y1[len(Y1)//2:])
    a = clf.score(X1[:len(X1)//2], Y1[:len(Y1)//2])
    
    return a,b
     
def noisevstest():
    a,b = makenu2(100,10,0.6)
    print(a,b)

def test_noise():
    N = 1000
    p_ave_list = []
    p_sd_list = []
    num_trails = 100
 
    noise_array = np.arange(0, 4, 0.2) 

    for noise in noise_array:
        p_list = []
        print(noise)
        count = 0
        while (count < num_trails):
            try:  
                a, b = makenu2(N, noise, 0.5)
                p_list.append(b)
                count = count + 1
            except ValueError:
                continue     
           
        p_ave = stat.mean(p_list)
        p_sd = stat.stdev(p_list)
        p_ave_list.append(p_ave)
        p_sd_list.append(p_sd)

    plt.errorbar(noise_array, p_ave_list, yerr = p_sd_list, 
                 fmt = 'o', color = 'r') 
    plt.title("Nu = 0.5")
    plt.xlabel('Noise')
    plt.ylabel('Accuracy')
    plt.show() 
       
def test_nu():
    N = 500
    p_ave_list = []
    p_sd_list = []
    num_trails = 100
 
    noise_array = np.arange(0.05, 0.5, 0.05) 

    for nu in noise_array:
        p_list = []
        print(nu)
        for i in range(num_trails):
            try:      
                a, b = makenu2(N, 1, nu)
                p_list.append(b)
            except ValueError:
                continue
        p_ave = stat.mean(p_list)
        p_sd = stat.stdev(p_list)
        p_ave_list.append(p_ave)
        p_sd_list.append(p_sd)

    plt.errorbar(noise_array, p_ave_list, yerr = p_sd_list, fmt = 'o', color = 'r') 
    plt.title("Noise = 1")
    plt.xlabel('Nu Value')
    plt.ylabel('Accuracy')
    plt.show()    
               
if __name__ == '__main__':
   makenu() 
   # makenu()
   # test_nu()
   #test_noise()
 #noisevstest() 
    #test_noise()
   #for i in range(20): 
        #try:
         #  test_nu()
       # except ValueError:
        #    continue
       
       
    