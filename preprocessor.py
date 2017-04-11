import numpy as np
from sklearn import preprocessing


data = np.array([[3,-1.5,2,-5.4],[0,4,-0.3,2.1],[1,3.3,-1.9,-4.3]])
data_standarized = preprocessing.scale(data)
print("\nMean = ",data_standarized.mean(axis=0))
print("Std Deviation",data_standarized.std(axis=0))


data_Scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled = data_Scaler.fit_transform(data)
print("\nData Scaled ",data_scaled)

data_normalized = preprocessing.normalize(data, norm='max')
print("\nData normalized",data_normalized)

data_binary = preprocessing.Binarizer(threshold=1.4).transform(data)
print("\nBinarized Data",data_binary)

encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print("\nEncoded Vector ",encoded_vector)
