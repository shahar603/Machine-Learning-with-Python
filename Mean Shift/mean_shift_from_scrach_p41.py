import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
import random


style.use('ggplot')



"""
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3],])
"""

n_centers = random.randint(2,5)


X,y = make_blobs(n_samples=50, centers=n_centers, n_features=2)

print(n_centers)


#plt.scatter(X[:,0], X[:,1], s=150)
#plt.show()




class MeanShift:

    def __init__(self, radius=None, radius_norm_step = 100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step



    def fit(self, data):

        if self.radius is None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm/self.radius_norm_step


        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        weights = range(self.radius_norm_step)[::-1]

        while True:
            new_centroids = []


            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]

                for featureset in data:
                    distance = np.linalg.norm(featureset - centroid)

                    if distance == 0:
                        distance = 10**-6

                    weight_index = int(distance/self.radius)

                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step - 1

                    to_add = weights[weight_index]*[featureset]
                    in_bandwidth += to_add



                    #if np.linalg.norm(featureset - centroid) < self.radius:
                    #    in_bandwidth.append(featureset)



                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))




            for i in uniques:
                for ii in uniques:
                    if i != ii and np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
                        uniques.remove(ii)
                        break


            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(prev_centroids[i], centroids[i]):
                    optimized = False
                    break

            if optimized:
                break

        self.centroids = centroids


        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []


        for featureset in data:
            distances = [np.linalg.norm(self.centroids[centroid] - featureset) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)




    def predict(self, data):
        distances = [np.linalg.norm(centroid - data) for centroid in self.centroids]
        return distances.index(min(distances))




colors = 10*['g', 'r', 'c', 'b', 'k']


clf = MeanShift()
clf.fit(X)


centroids = clf.centroids



for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()