import numpy as np
import mglearn as mglearn
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

x,y = mglearn.datasets.make_wave(n_samples = 40)


fig, axes = plt.subplots(1,3,figsize = (15,4))
line = np.linspace(-3,3,1000).reshape(-1,1)

plt.suptitle("Nearest_Neighbors_Regression")

for n_neighbors, ax in zip([1,3,9],axes):
	reg = KNeighborsRegressor(n_neighbors = n_neighbors).fit(x,y)
	ax.plot(x,y,'o')
	ax.plot(x,-3*np.ones(len(x)),'o')
	ax.plot(line,reg.predict(line))
	ax.set_title("%d neighbor(s) " %n_neighbors)

plt.show()
