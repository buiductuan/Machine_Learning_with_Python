import mglearn as mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


x,y = mglearn.datasets.make_wave(n_samples = 60)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42)

lr = LinearRegression().fit(x_train,y_train)

mglearn.plots.plot_linear_regression_wave()

print("lr.coef_ : %s" % lr.coef_)

print("lr.intercept_ : %s" % lr.intercept_)

print("training set score : %f" %lr.score(x_train,y_train))
print("testing set score : %f" %lr.score(x_test,y_test))