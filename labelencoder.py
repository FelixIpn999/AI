from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
input_classes = ['audi', 'ford', 'audi', 'toyota', 'nissan']
label_encoder.fit(input_classes)


print("\Label Mapping")
for i , item in enumerate (label_encoder.classes_):
    print(item, '----->', i)
labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print("\nLabels", labels)
print("\nEncoded Labels", encoded_labels)

encoded = [2, 1, 0, 3, 1]
decoded_labels= label_encoder.inverse_transform(encoded)
print("\nEncoded labels", encoded)
print("\nDecoded Labels", decoded_labels)