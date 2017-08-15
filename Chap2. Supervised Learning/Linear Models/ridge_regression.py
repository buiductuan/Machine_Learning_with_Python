import mglearn as mglearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

x,y= mglearn.datasets.load_extended_boston()

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)

lr = LinearRegression().fit(x_train,y_train)

ridge = Ridge().fit(x_train,y_train)
ridge10 = Ridge(alpha = 10).fit(x_train,y_train)
# ridge100 = Ridge(alpha = 100).fit(x_train,y_train)
ridge01 = Ridge(alpha = .1).fit(x_train,y_train)

lasso = Lasso().fit(x_train,y_train)

# lr_only_ridge = Ridge()

# print(lr_only_ridge)
print("Training set score of LR : %f" %lr.score(x_train,y_train))
print("Testing set score of LR :%f" %lr.score(x_test,y_test))

print("Training set score of alpha = 1 : %f" %ridge.score(x_train,y_train))
print("Testing set score of alpha = 1 : %f" %ridge.score(x_test,y_test))

print("Training set score of alpha = 10 : %f" %ridge10.score(x_train,y_train))
print("Testing set score of alpha = 10 : %f" %ridge10.score(x_test,y_test))


print("Training set score of alpha = 0.1 : %f" %ridge01.score(x_train,y_train))
print("Testing set score of alpha = 0.1 : %f" %ridge01.score(x_test,y_test))

print("Training set score of LR : %f" %lr.score(x_train,y_train))
print("Testing set score of LR :%f" %lr.score(x_test,y_test))


plt.title("Ridge_coefficients")
plt.plot(ridge.coef_,'o',label = "Ridge alpha = 1")
plt.plot(ridge10.coef_,'o',label = "Ridge alpha = 10")
plt.plot(ridge01.coef_,'o',label = "Ridge alpha = 0.1")
plt.plot(lr.coef_,'o',label = "linear Regression")
plt.plot(lasso.coef_,'o',label = "Lasso alpha = 1")
plt.ylim(-25,25)
plt.legend()

plt.show()

# print("================================== X ==================================")
# print(x)
# print("================================== Y ==================================")
# print(y)
# print("================================== lr ==================================")
# print(lr)
# print("================================== x_train ==================================")
# print(x_train)
# print("================================== y_train ==================================")
# print(y_train)
# print("================================== x_test ==================================")
# print(x_test)
# print("================================== y_test ==================================")
# print(y_test)