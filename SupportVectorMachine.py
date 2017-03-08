from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt 
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)

#SVM for Four Class dataset (may not be the best dataset to use this on, but I learned how to use SVMs)
f = open("fourclass.txt",'r')
x = []
xIdx = []
y = []

#clean up the data
for line in f: 
	line = line.replace("+","")
	line = line.replace("1:","")
	line = line.replace("2:","")

	line = line.split(' ')

	#make label list
	if line[0] == '-1':
		y.append(0)
	else:
		y.append(1)
	
	#make data pts 
	xIdx.append(float(line[1]))
	xIdx.append(float(line[2]))

	#add data pts to list 
	x.append(xIdx)

	xIdx = []

#convert data pt list to numpy array
X = np.array(x)

#train the SVM
clf = svm.SVC(kernel='linear')
clf.fit(X,y)

#predict where a new pt will be classified based on SVM 
print("Predicted cluster will be:")
print(clf.predict([111.0,101.0]))

#get stuff to plot the data and a line
w = clf.coef_[0]
a = -w[0] / w[1]

xx = np.linspace(0,300)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx,yy,'k-')
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.show()

f.close()





