# Python-and-R
Python And R

# Linear Regression with Diabetes from sklearn

**In this article,**


 we will learn about the linear regression algorithm with examples. First, we will understand the basics of linear regression algorithm, and then we will look at the steps involved in linear regression and finally an example of linear regression.

Regression is a supervised learning technique for determining the relationship between two or more variables. “Regression fits a line or curve that passes through all the data points on a target-predictor graph in such a way that the vertical distance between the data points and the regression line is minimum”.  Regression is mainly used for prediction, time series analysis, forecasting, etc. There are many types of regression algorithms like linear regression, multiple linear regression, logistic regression, and polynomial regression.

Linear regression is a statistical method that is used for prediction based on the relationship between the continuous variables. In simple words, we can say that linear regression shows the linear relationship between the independent variable (X-axis) and the dependent variable (Y-axis), consequently called linear regression. If there is a single input variable (x), such linear regression is called simple linear regression. And if there is more than one input variable, such linear regression is called multiple linear regression.

The linear regression model depicts the relationship between the variables as a sloped straight line as shown in the graph below. When the value of x (independent variable) increases, the value of y (dependent variable) is likewise increasing. In linear regression what we do is find a best fit straight line similar to the red line shown in the graph that fits the given data points best (i.e. with minimum error).

Load Python library packages
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
```
```
#load the diabets dataset from sklearn
diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)
```

```
#use only feature
diabetes_x = diabetes_x[:, np.newaxis, 2]
```
'''
#split the targets into training/testing sets
diabetes_x_train = diabetes_x[:-20]
diabetes_x_test = diabetes_x[-20:]
'''
```
#split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
```

```
#now let's creat linear regression
regr = linear_model.LinearRegression()
```
```
#Train the model using the traing sets
regr.fit(diabetes_x_train, diabetes_y_train)
```

```
#Make predictions usong the testing sets
diabetes_y_pred =regr.predict(diabetes_x_test)
```

```
#the coefficients
print('coefficients:n', regr.coef_)

#mean mean squred error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))


#the coefficients of determinationa: 1 is pefect prediction

print('coefficients of determination: %.2f'
       % r2_score(diabetes_y_test, diabetes_y_pred))

#print output
plt.scatter(diabetes_x_test, diabetes_y_test, color='black')
plt.plot(diabetes_x_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

```

![image](https://github.com/JoshuaKab/Python-and-R/assets/135429439/ab4bc9aa-e193-4d9e-ba7d-a409d8690af5)




