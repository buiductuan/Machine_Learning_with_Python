import mglearn as ml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


x,y = ml.datasets.make_forge()

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

clf = KNeighborsClassifier(n_neighbors = 3)

clf.fit(x_train,y_train)

KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',
					metric_params=None,n_jobs=1,n_neighbors=3,p=2,weights='uniform')
print('',clf.predict(x_test))

print('',clf.score(x_test,y_test))