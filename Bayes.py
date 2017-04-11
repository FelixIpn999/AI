import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from utilities import visualize_classifier

input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file,delimiter=',')
X , y = data[:,:-1],data[:,:-1]

classifier = GaussianNB()
classifier.fit(X,y)


y_pred = classifier.predict(X)

accuracy = 100*(y== y_pred).sum()/X.shape[0]

print("Accuracy of the new classifier =", round(accuracy, 2), "%")

visualize_classifier(classifier,X,y)



