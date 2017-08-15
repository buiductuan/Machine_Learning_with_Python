import mglearn as mglearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


x,y = mglearn.datasets.load_extended_boston()

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)


lr = LinearRegression().fit(x_train,y_train)

print("Training set score : %f" %lr.score(x_train,y_train))
print("Test set score : %f" %lr.score(x_test,y_test))