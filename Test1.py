import numpy as np
from sklearn import preprocessing

input_labels = ['red','black','red','green','black','yellow','white']
labels_encoder = preprocessing.LabelEncoder()
labels_encoder.fit(input_labels)
#Print the mapping
print("\nMapping\n")
for i , item in enumerate(labels_encoder.classes_):
    print(item,'--->',i)

#Encode
test_labels = ['green','red','black']
encode_values = labels_encoder.transform(test_labels)
print("\nLabels:",test_labels)
print("\nEncoded Values",list(encode_values))

#DECODE
encoded_values = [3,0,4,1]
decoded_list = labels_encoder.inverse_transform(encode_values)
print("\nEncoded Values",encode_values)
print("\Decoded Values",decoded_list)
