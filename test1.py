import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# linear data
X = np.array([5., 6., 7., 2., 2., 1.])
y = np.array([5., 4., 8., 4., 3., 2.])

train_X = np.vstack((X, y)).T
train_y = [1, 1, 1, -1, -1, -1]

clf = svm.SVC(kernel='linear', C=1.0)
model = clf.fit(train_X, train_y)

print(model.coef_, model.intercept_)

# get the weight values for the linear equation from the trained SVM model
w = model.coef_[0]

# get the y-offset for the linear equation
a = -w[0] / w[1]

# make the x-axis space for the data points
XX = np.linspace(0, 8)

# x2= -w1/w2 * x1 - b/w2
yy = a * XX - model.intercept_[0] / w[1]

# plot the decision boundary
plt.plot(XX, yy, 'k-')

# show the plot visually
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y)
plt.show()

X = np.array([5., 6., 7., 2., 2., 1., 3., 4.])
y = np.array([5., 4., 8., 4., 3., 2., 4., 5.])

train_X = np.vstack((X, y)).T
train_y = [1, 1, 1, -1, -1, -1, 1, -1]

clf = svm.SVC(kernel='linear', C=0.01)
model = clf.fit(train_X, train_y)

print(model.coef_, model.intercept_)

w = model.coef_[0]
a = -w[0] / w[1]

XX = np.linspace(0, 8)

# x2= -w1/w2 * x1 - b/w2
yy = a * XX - (model.intercept_[0]) / w[1]

plt.plot(XX, yy, 'k-')
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y)
plt.show()
