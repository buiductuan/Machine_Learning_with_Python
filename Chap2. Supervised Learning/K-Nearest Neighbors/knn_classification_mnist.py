import os
import numpy as np
from mnist import MNIST

import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import time


mndata  = MNIST(os.getcwd()+"/data/MNIST/")

mndata.load_testing()
mndata.load_training()

x_test = mndata.test_images
x_train = mndata.train_images

y_test = np.asarray(mndata.test_labels)
y_train = np.asarray(mndata.train_labels)

start_time = time.time()

clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p=2)

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

end_time = time.time()

print ("accuracy of 1NN for MNIST : % 2.f%%" %(100*accuracy_score(y_test,y_pred)))

print("Running time : %d" %(end_time - start_time))
