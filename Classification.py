import numpy as np
from sklearn import preprocessing

input_Data = np.array( [[5.1,-2.9,3.3],[-1.2,7.8,-6.1],[3.9,0.4,2.1],[7.3,-9.9,-4.5]] )
#Binarize#
data_binarized= preprocessing.Binarizer(threshold=-0.1).transform(input_Data)
print("\nBinarized Data\n" ,data_binarized)

#Mean Removal
print("\nBefore\n")
print("\nMean = \n",input_Data.mean(axis=0))
print("\nStd Deviation\n ",input_Data.std(axis=0))
#Remove Mean
data_scaled = preprocessing.scale(input_Data)
print("\nAFTER\n")
print("\nMean\n",data_scaled.mean(axis=0))
print("\nStd Deviation\n",data_scaled.std(axis=0))

#Min-Max Scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaler_minmax= data_scaler_minmax.fit_transform(input_Data)
print("\nMin Max\n ",data_scaler_minmax)

#Normalize
data_normalized = preprocessing.normalize(input_Data,norm='l1')
data_normalized_2 = preprocessing.normalize(input_Data,norm='l2')
print("\nL1 Normalized\n",data_normalized)
print("\nL2 Normalized\n",data_normalized_2)

