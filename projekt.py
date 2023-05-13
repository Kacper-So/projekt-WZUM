import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

dataset = []
with open('sample_dataset.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset.append(row)

X = pd.DataFrame(dataset)
X = X.drop(columns=[''])
Y = X['letter']
X = X.drop(columns=['letter'])
# X = X.drop(columns=['landmark_0.x', 'landmark_0.y', 'landmark_0.z','landmark_1.x', 'landmark_1.y', 'landmark_1.z', 'landmark_2.x', 'landmark_2.y', 'landmark_2.z','landmark_3.x', 'landmark_3.y', 'landmark_3.z','landmark_4.x', 'landmark_4.y', 'landmark_4.z', 'landmark_5.x', 'landmark_5.y', 'landmark_5.z', 'landmark_6.x', 'landmark_6.y', 'landmark_6.z','landmark_7.x', 'landmark_7.y', 'landmark_7.z', 'landmark_8.x', 'landmark_8.y', 'landmark_8.z','landmark_9.x', 'landmark_9.y', 'landmark_9.z','landmark_10.x', 'landmark_10.y', 'landmark_10.z', 'landmark_11.x', 'landmark_11.y', 'landmark_11.z', 'landmark_12.x', 'landmark_12.y', 'landmark_12.z','landmark_13.x', 'landmark_13.y', 'landmark_13.z', 'landmark_14.x', 'landmark_14.y', 'landmark_14.z','landmark_15.x', 'landmark_15.y', 'landmark_15.z','landmark_16.x', 'landmark_16.y', 'landmark_16.z', 'landmark_17.x', 'landmark_17.y', 'landmark_17.z', 'landmark_18.x', 'landmark_18.y', 'landmark_18.z','landmark_19.x', 'landmark_19.y', 'landmark_19.z', 'landmark_20.x', 'landmark_20.y', 'landmark_20.z'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

model = svm.SVC(kernel='linear')

model.fit(X_train, Y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, pred)
print(accuracy)