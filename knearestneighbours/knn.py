import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

'''
Goal:
    Take a graph and plot a set of data points that are classified either blue or red
    Take a new point and try to classify the point as either red or blue

Step 1
    Take the Euclidean distance between the new point and all the remaining points on the
    graph

    #Distance between new point and another one point on the graph
    new point = (xi,yi)
    another point = (xf,yf)

    Euclidean dist. = ((xf-xi)^2 + (yf-yi)^2)^(1/2)

    i = initial
    f = final

Step 2
    Take the k smallest distances and determine how many of those points are either
    red or blue

    Classify the new point as red if there are more red points closer
    Classify the new point as blue if there are more blue points closer

'''

'''
points = {
    "blue" : [[2,4], [1,3], [2,3], [3,2], [2,1]],
    "red" : [[5,6], [4,5], [4,6], [6,6], [5,4],]
}

newPoint = [5,5]

def EuclideanDist(i, f):
    #Calculates Distance from new point to final point
    return np.sqrt(np.sum((np.array(f) - np.array(i))**2))

class KNN:
    def __init__ (self, k=3):
        self.k = k
        self.points = None

    def fit(self, points):
        #training data essentially, assign the model the points on the graph
        self.points = points

    def predict(self, newPoint):
        distances = []

        #category is either blue or red
        for category in self.points:
            for point in self.points[category]:
                distance = EuclideanDist(point, newPoint)
                distances.append([distance, category])
                #append the distance from the point and whether that point is a blue or red
        
        #take the k most frequent points
        mostFrequentPointsColors = [category[1] for category in sorted(distances)[:self.k]]

        #find the most common color
        result = Counter(mostFrequentPointsColors).most_common(1)[0][0]
        return result
    
classifier = KNN()
classifier.fit(points)
classifier.predict(newPoint)

#Visualize

ax = plt.subplot()
ax.grid(True, color="#323232")
ax.set_facecolor("black")
ax.figure.set_facecolor("#121212")
ax.tick_params(axis="x", color="white")
ax.tick_params(axis="y", color="white")

for point in points["blue"]:
    ax.scatter(point[0], point[1], color="#104DCA", s=60)

for point in points["red"]:
    ax.scatter(point[0], point[1], color="#FF0000", s=60)

prediction = classifier.predict(newPoint)

color = "#FF0000" if prediction == "red" else "#104DCA"
ax.scatter(newPoint[0], newPoint[1], color=color, marker="*", s=200, zorder=100)

for point in points["blue"]:
    ax.plot([newPoint[0], point[0]],[newPoint[1], point[1]],color="#104DCA", linestyle="--", linewidth = 1)

for point in points["red"]:
    ax.plot([newPoint[0], point[0]],[newPoint[1], point[1]],color="#FF0000", linestyle="--", linewidth = 1)

plt.show()


#Below is the code for 3D KNN
'''

points = {
    "blue" : [[2,4,3], [1,3,5], [2,3,1], [3,2,3], [2,1,6]],
    "red" : [[5,6,5], [4,5,2], [4,6,1], [6,6,1], [5,4,6], [10,10,4]]
}

newPoint = [7,7,7]

def EuclideanDist(i, f):
    #Calculates Distance from new point to final point
    return np.sqrt(np.sum((np.array(f) - np.array(i))**2))

class KNN:
    def __init__ (self, k=3):
        self.k = k
        self.points = None

    def fit(self, points):
        #training data essentially, assign the model the points on the graph
        self.points = points

    def predict(self, newPoint):
        distances = []

        #category is either blue or red
        for category in self.points:
            for point in self.points[category]:
                distance = EuclideanDist(point, newPoint)
                distances.append([distance, category])
                #append the distance from the point and whether that point is a blue or red
        
        #take the k most frequent points
        mostFrequentPointsColors = [category[1] for category in sorted(distances)[:self.k]]

        #find the most common color
        result = Counter(mostFrequentPointsColors).most_common(1)[0][0]
        return result
    
classifier = KNN()
classifier.fit(points)
classifier.predict(newPoint)

#Visualize

fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(projection="3d")
ax.grid(True, color="#323232")
ax.set_facecolor("black")
ax.figure.set_facecolor("#121212")
ax.tick_params(axis="x", color="white")
ax.tick_params(axis="y", color="white")

for point in points["blue"]:
    ax.scatter(point[0], point[1], point[2], color="#104DCA", s=60)

for point in points["red"]:
    ax.scatter(point[0], point[1], point[2], color="#FF0000", s=60)

prediction = classifier.predict(newPoint)
color = "#FF0000" if prediction == "red" else "#104DCA"
ax.scatter(newPoint[0], newPoint[1], newPoint[2], color=color, marker="*", s=200, zorder=100)

for point in points["blue"]:
    ax.plot([newPoint[0], point[0]],[newPoint[1], point[1]],[newPoint[2], point[2]] ,color="#104DCA", linestyle="--", linewidth = 1)

for point in points["red"]:
    ax.plot([newPoint[0], point[0]],[newPoint[1], point[1]], [newPoint[2], point[2]], color="#FF0000", linestyle="--", linewidth = 1)

plt.show()

