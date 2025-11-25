import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

imaage_as_array = mpimg.imread('palm.jpg')
print(imaage_as_array)
print(len(imaage_as_array))

plt.imshow(imaage_as_array)
plt.show()


#2D
(h,w,c) = imaage_as_array.shape
imaage_as_array_2d = imaage_as_array.reshape(h,w,c)
print(imaage_as_array_2d)
print(len(imaage_as_array_2d))
from sklearn.cluster import  kmeans
model =kmeans(n_clusters=4)
labels = model.fit_predict(imaage_as_array_2d)
print(labels)
print(model.cluster_centers_)