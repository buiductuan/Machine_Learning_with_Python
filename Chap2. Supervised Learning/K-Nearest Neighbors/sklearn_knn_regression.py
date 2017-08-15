import mglearn as mglearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

x,y = mglearn.datasets.make_wave(n_samples = 40)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

reg = KNeighborsRegressor(n_neighbors = 3)

reg.fit(x_train,y_train)

KNeighborsRegressor(algorithm = 'auto' , leaf_size = 30,
					metric='minkowski',metric_params= None,n_jobs=1,
					n_neighbors = 3 ,p=2,weights='uniform')

print("Du lieu kiem thu : ",reg.predict(x_test))

print("Danh gia : %f%%" %(100*reg.score(x_test,y_test)))
