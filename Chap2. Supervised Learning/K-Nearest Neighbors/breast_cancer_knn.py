import mglearn as mglearn
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split




cancer = load_breast_cancer()

# print("======================== DATASETS BREAST CANCER =================================")
# print("Number of data points : %d" %len(cancer.target))
# print("=================================================================================")

x_train,x_test,y_train,y_test = train_test_split(cancer.data,cancer.target,
	stratify=cancer.target,random_state=66)

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
	clf = KNeighborsClassifier(n_neighbors=n_neighbors)
	clf.fit(x_train,y_train)

	training_accuracy.append(clf.score(x_train,y_train))

	test_accuracy.append(clf.score(x_test,y_test))

plt.plot(neighbors_settings,training_accuracy,label="training accuracy")
plt.plot(neighbors_settings,test_accuracy,label="test accuracy")

plt.legend() #show name line
plt.show()

# print (test_accuracy)
# print('\n')
# print (training_accuracy)