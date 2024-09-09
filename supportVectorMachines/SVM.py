import numpy as np
import matplotlib as plt
from sklearn import datasets

'''

Goal:
    To find the best function (lin/hyperplane) that separates two clusters or class
    of data points

    distance from hyperplane and nearest point of each class creates a margin
    margin should be 50 - 50 on each side of the hyperplane function 
    (max distance on both sides of the hyperplane)

    Step 1 - Linear Model
        w = weights vector
        b = bias vector
        x = x vector

        Linear Model
            (x dot w) + b //this is our line
        
        class a = 1;
        class b = -1;
        yi = either class a or class b (1 or -1)
            
        say you are predicting a point (x1,x2) which is of class yi
        draw vector x from origin to point
        draw vector perpindicular to linear model from the origin (w vector)
        - ||w|| = c
        - if magnitude of x vector in direction of w vector 
            = c //point is on line
            < c //point is of class -1
            > c //point is of class 1
        
        w^hat = w vector / ||w||
        Magnitude of x vector in direction of w vector  = proj of x vector across w vector
            x dot w = c // point is on the line
            x dot w < c // point is of class -1
            x dot w > c // point is of class 1

        We can rewrite expression x dot w - c = 0 and plug in b for -c
        hence 
        yi = 1 if x dot w + b >= 0
        yi = -1 if x dot w +b < 0

        yi represents the class 1 or -1

        We can rewwrite=
        yi(w dot x + b) >= 1
        yi * f(x)
        plug in x values and the class its supposed to be in yi, if the condition is met
        then the model is working
    
        Distance 
        draw any vector from one point on margin to the other margin (vector P)
        draw perpindicular vector w to the margins vector (w)
        and find the vector projection of P on to the direction vector of w

        You should get that 2/||w|| = d so we have to make this a maximum

    Step 3 Hinge Loss
        
        error = max(0, 1- yi * f(x))
        error is 0 otherwise it is equal to the negative value

        - Note we want to minimize the loss error while maximizing the distance between margins

        max(2/||w||) = min(0.5||w||^2)

        Loss = 0.5||w||^2 - summation(error)

            Loss = 0.5||w||^2                       // if there is no error
            Loss = 0.5||w||^2 + 1 - yi(w dot x - b) // if there is error

    Step 4 Gradient Descent

        if yi * f(x) >= 1
            dL/dw = w
            dL/db = 0

        else
            dL/dw = w - yi - x
            dL/db = yi

        w new = w - learningRate * dw
        b new = b - learningRate * db

    Step 5 Soft Margins
        Sometimes there are outliers within the data so this is so that the model
        allows k number of points to be classified wrong

        0.5||w||^2  - summation(error) + c * summation(zeta)

        SVM Error = Margin Error + Classification Error
        high value of c would mean that you dont want to focus on margin error at all
    
        
    Step 6 Kernel Functions
        Some data points we can not draw any single line through the data so we use the kernel trick
        This is to lift the points up to another dimension and then splitting with a hyperplane

        Some Kernel Functions to use

        Polynomial Kernel Function
            f(x,y) = (x dot y + 1)^d
            d = degree

        Sigmoid Kernel Function
            f(x, y) = tanh*(kx dot y + x)

        RBF Kernel Function
            f(x,y) = e^((-||x-y||^2)/(2k^2))

'''

class SupportVectorMachine:

    def __init__(self, learningRate=0.001, lambdaParam=0.01, epochs=1000):
        self.learningRate = learningRate #learning Rate
        self.lambdaParam = lambdaParam #lambda parameter
        self.epochs = epochs #number of iterations
        self.w = None
        self.b = None
        
    def fit(self, x, y):
        # if y is less than equal to zero change it to -1 else change it to 1
        y = np.where(y <= 0, -1, 1)

        #input vector x has number of rows = number of samples and number of columns is number of features
        nSamples, nFeatures = x.shape

        #for w create a 0 vector of size nFeatures
        self.w = np.zeros(nFeatures)
        self.b = 0

        #Gradient Descent 
        for _ in range(self.epochs): #iterate over epochs
            for i, xi in enumerate(x): #gives index and current sample
                condition = y[i] * (np.dot(xi, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learningRate * (2 * self.lambdaParam * self.w)
                else:
                    self.w -= self.learningRate * (2 * self.lambdaParam * self.w - np.dot(xi, y[i]))
                    self.b -= self.learningRate * y[i]
        
    def predict(self, x):
        #we want to take the dot product of x and w
        linearOutput = np.dot(x, self.w) - self.b
        
        #return the sign the output so it is either Class 1 or Class -1
        return np.sign(linearOutput) # 1 or -1 can also return 0 if 0

    
