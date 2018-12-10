import matplotlib.pyplot as plt
from sklearn import datasets, svm

digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=100)

x,y = digits.data[:-4], digits.target[:-4]
clf.fit(x,y)

print('Prediction:', clf.predict(digits.data[[-2]]))
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
