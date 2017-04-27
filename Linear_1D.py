import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

def main():

    data = pd.read_csv('LR_1D.txt', header = None, names = ['population', 'profit'])
    data = data.as_matrix()

    X = []
    Y = []

    for line in data:
        x, y = line
        X.append(x)
        Y.append(y)
        
    X = np.array(X)
    Y = np.array(Y)   

    #Visualization
    plt.scatter(X,Y)
    plt.xlim(4,24)
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()


    #Fit
    denominator = X.dot(X) - X.mean() * X.sum()
    w = (X.dot(Y) - Y.mean() * X.sum()) / denominator
    b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator


    Yhat = w * X + b

    #Visualization with line
    plt.scatter(X,Y)
    plt.plot(X,Yhat, c='r')
    plt.xlim(4,24)
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()


    #R-Squared to measure accuracy
    d1 = Y - Yhat 
    d2 = Y - Y.mean()

    #sum of squares residual divided by sum of squares total
    R2 = 1 - d1.dot(d1) / d2.dot(d2)

    print ("The R-squared is:", R2)



    #Using Sklearn to create same results

    X = X.reshape(-1,1)

    regr = LinearRegression()
    regr.fit(X,Y)

    plt.scatter(X,Y)
    xx = np.arange(5,24)
    plt.plot(xx,regr.coef_*xx + regr.intercept_, c='g')
    plt.xlim(4,24)
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()


if __name__ == '__main__':
    main()

