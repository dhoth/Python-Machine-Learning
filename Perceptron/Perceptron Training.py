import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import Perceptron

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = None, names = ["Sepal",2,"Petal",4,"class"])
#print(df.tail)

#select setosa and versicolor - your classes
Y = df.loc[0:99,"class"].values
Y = np.where(Y == "Iris-setosa", -1,1)

#extract the sepal length and petal length
X = df.loc[0:99,["Sepal","Petal"]].values

#plot the data
plt.scatter(X[:50,0], X[:50,1], color = "red", marker = "o", label = "setosa")
plt.scatter(X[50:100,0], X[50:,1], color = "blue", marker = 'x', label = "versicolor")
plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
plt.legend(loc = "upper left")
#plt.show()

#train the Perceptron
p = Perceptron(eta = .1, n_iter = 10)
p.fit(X,Y)
#plt.plot(range(1,len(p.errors_)+1), p.errors_,marker = 'o')
#plt.xlabel("Epochs")
#plt.ylabel("Number of Updates")
#plt.show()



x_min = np.amin(X,axis = 0)[0]
x_max = np.amax(X,axis = 0)[0]

x = np.linspace(x_min,x_max,1000)
y = [(-p.w_[0]-p.w_[1]*i)/p.w_[2] for i in x]
plt.plot(x,y, color = "black")
plt.show()
#x = np.linspace()

#Show that it can't seperate XOR
X = np.array([[0,0],[1,1],[1,0],[0,1]])
Y = np.array([-1,-1,1,1])



#plt.scatter(X[:2,0], X[:2,1], color = "red", marker = "x", label = "TRUE")
#plt.scatter(X[2:4,0], X[2:4,1], color = 'blue', marker = "o", label = "FALSE")

p.fit(X,Y)
#plt.plot(range(1,len(p.errors_)+1), p.errors_,marker = 'o')
#plt.xlabel("Epochs")
#plt.ylabel("Number of Updates")


plt.show()
#x_min = np.amin(X,axis = 0)[0]
#x_max = np.amax(X,axis = 0)[0]

#x = np.linspace(x_min,x_max,1000)
#y = [(-p.w_[0]-p.w_[1]*i)/p.w_[2] for i in x]
#plt.plot(x,y)
plt.show()
