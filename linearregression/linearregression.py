import pandas as pd 
import matplotlib.pyplot as plt

''' 

In this Linear Regression we want to find the best fit line to a set of data points
Goal: find correct Values of m and b to find a best fit line

m = weight
b = bias

Step 1 
    Calculate the Mean Squared Error between each point on the line and each data point
    at a certain x value
    
    n = number of data points
    yi = ith data point's y value
    xi = ith data point's x value
    mx+b = best fit line

    ~ E = 1/n * Summation( (yi - (m(xi)+b))^2 )
    //summation of all errors squared divided by number of points

Step 2
    We want to minimize E (Error) as much as possible
    - Find how to Maximize E and take the opposite
    - take partial derivative of E with respect to m
    - take partial derivative of E with respect to b

    dE/dm = 1/n * Summation(2*(yi  - (m(xi)+b)) * -xi)
          = -2/n * Summation(xi * (yi - (m(xi)+b)))

    dE/db = -2/n * Summation(yi - (m(xi)+b))

Step 3
    Take current weight and bias and subtract by learning rate * maximimum of the 
    partial derivative 

    L = Learning Rate - how large the steps are we take

    mf = mi - L * dE/dm
    bf = bi - L * dE/db

'''

data = pd.read_csv('linearregression.csv') 


#function for calculating mean Squared Error
#Not used in code since Gradient Descent Function implements this
def MeanSquaredError(m, b, points):
    totalError = 0
    for i in range(len(points)):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        totalError += y - (m*x + b) ** 2
    totalError / float(len(points))

def GradientDescent(m_curr, b_curr, points, lr):
    mGradient = 0
    bGradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        mGradient += -(2/n) * x * (y-(m_curr * x + b_curr))
        bGradient += -(2/n) * (y-(m_curr * x + b_curr))

    m = m_curr - mGradient * lr
    b = b_curr - bGradient * lr
    return m, b

m = -100
b = 120000
learningRate = 0.001
epochs = 7000 #number of iterations of gradient descents

for i in range(epochs):
    if i%100 == 0:
        print(f"epoch: {i}")
        plt.scatter(data.YearsExperience, data.Salary, color="black")
        plt.plot(list(range(0,12)), [m * x + b for x in range(0,12)], color ="red")
        plt.show()

    m, b = GradientDescent(m,b,data,learningRate)

print(m,b)
#plt.scatter(data.YearsExperience, data.Salary, color="black")
#plt.plot(list(range(0,12)), [m * x + b for x in range(0,12)], color ="red")
#plt.show()

