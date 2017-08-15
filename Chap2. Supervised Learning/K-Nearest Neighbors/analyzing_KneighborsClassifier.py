import mglearn as ml
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from sklearn.neighbors import KNeighborsClassifier


x,y = ml.datasets.make_forge()


fig,axes = plt.subplots(1,3,figsize=(10,3))

for n_neighbors , ax in zip([1,3,9],axes):
	
	clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(x,y)
	
	ml.plots.plot_2d_separator(clf , x , fill=True , eps=0.5 , ax=ax , alpha=.4)
	
	ax.scatter(x[:,0] , x[:,1] , c=y , s=60 , cmap=ml.cm2)
	
	ax.set_title("%d neighbor(s) " % n_neighbors)

plt.show()