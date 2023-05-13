import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix

dataset = []
with open('dataset.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset.append(row)

with open('sample_dataset.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset.append(row)

X = pd.DataFrame(dataset)
X = X.drop(columns=[''])

fingers = ['1', '2', '3', '4', '5']
finger_1 = []
finger_2 = []
finger_3 = []
finger_4 = []
finger_5 = []

for i in range(0, len(X)):
    for finger in fingers:
        if finger == '1':
            if float(X['landmark_4.x'][i]) > float(X['landmark_17.x'][i]): #kciuk z lewej
                if float(X['world_landmark_4.x'][i]) > float(X['world_landmark_5.x'][i]):
                    finger_1.append('1')
                elif float(X['world_landmark_4.x'][i]) > float(X['world_landmark_9.x'][i]) and float(X['world_landmark_4.x'][i]) < float(X['world_landmark_5.x'][i]):
                    finger_1.append('2')
                elif float(X['world_landmark_4.x'][i]) > float(X['world_landmark_13.x'][i]) and float(X['world_landmark_4.x'][i]) < float(X['world_landmark_9.x'][i]):
                    finger_1.append('3')
                else:
                    finger_1.append('4')
            else: #kciuk z prawej
                if float(X['world_landmark_4.x'][i]) < float(X['world_landmark_5.x'][i]):
                    finger_1.append('1')
                elif float(X['world_landmark_4.x'][i]) < float(X['world_landmark_9.x'][i]) and float(X['world_landmark_4.x'][i]) > float(X['world_landmark_5.x'][i]):
                    finger_1.append('2')
                elif float(X['world_landmark_4.x'][i]) < float(X['world_landmark_13.x'][i]) and float(X['world_landmark_4.x'][i]) > float(X['world_landmark_9.x'][i]):
                    finger_1.append('3')
                else:
                    finger_1.append('4')

        elif finger == '2':
            ref_dist = math.sqrt(pow(float(X['world_landmark_5.x'][i]) - float(X['world_landmark_0.x'][i]), 2) + pow(float(X['world_landmark_5.y'][i]) - float(X['world_landmark_0.y'][i]), 2) + pow(float(X['world_landmark_5.z'][i]) - float(X['world_landmark_0.z'][i]), 2))
            finger_tip_dist = math.sqrt(pow(float(X['world_landmark_8.x'][i]) - float(X['world_landmark_0.x'][i]), 2) + pow(float(X['world_landmark_8.y'][i]) - float(X['world_landmark_0.y'][i]), 2) + pow(float(X['world_landmark_8.z'][i]) - float(X['world_landmark_0.z'][i]), 2))
            th1 = 1.2
            th2 = 0.8
            if finger_tip_dist > th1 * ref_dist:
                finger_2.append('straight')
            elif finger_tip_dist < th1 * ref_dist and finger_tip_dist > th2 * ref_dist:
                finger_2.append('bent')
            else:
                finger_2.append('closed')

        elif finger == '3':
            ref_dist = math.sqrt(pow(float(X['world_landmark_9.x'][i]) - float(X['world_landmark_0.x'][i]), 2) + pow(float(X['world_landmark_9.y'][i]) - float(X['world_landmark_0.y'][i]), 2) + pow(float(X['world_landmark_9.z'][i]) - float(X['world_landmark_0.z'][i]), 2))
            finger_tip_dist = math.sqrt(pow(float(X['world_landmark_12.x'][i]) - float(X['world_landmark_0.x'][i]), 2) + pow(float(X['world_landmark_12.y'][i]) - float(X['world_landmark_0.y'][i]), 2) + pow(float(X['world_landmark_12.z'][i]) - float(X['world_landmark_0.z'][i]), 2))
            th1 = 1.2
            th2 = 0.8
            if finger_tip_dist > th1 * ref_dist:
                finger_3.append('straight')
            elif finger_tip_dist < th1 * ref_dist and finger_tip_dist > th2 * ref_dist:
                finger_3.append('bent')
            else:
                finger_3.append('closed')

        elif finger == '4':
            ref_dist = math.sqrt(pow(float(X['world_landmark_13.x'][i]) - float(X['world_landmark_0.x'][i]), 2) + pow(float(X['world_landmark_13.y'][i]) - float(X['world_landmark_0.y'][i]), 2) + pow(float(X['world_landmark_13.z'][i]) - float(X['world_landmark_0.z'][i]), 2))
            finger_tip_dist = math.sqrt(pow(float(X['world_landmark_16.x'][i]) - float(X['world_landmark_0.x'][i]), 2) + pow(float(X['world_landmark_16.y'][i]) - float(X['world_landmark_0.y'][i]), 2) + pow(float(X['world_landmark_16.z'][i]) - float(X['world_landmark_0.z'][i]), 2))
            th1 = 1.2
            th2 = 0.8
            if finger_tip_dist > th1 * ref_dist:
                finger_4.append('straight')
            elif finger_tip_dist < th1 * ref_dist and finger_tip_dist > th2 * ref_dist:
                finger_4.append('bent')
            else:
                finger_4.append('closed')

        elif finger == '5':
            ref_dist = math.sqrt(pow(float(X['world_landmark_17.x'][i]) - float(X['world_landmark_0.x'][i]), 2) + pow(float(X['world_landmark_17.y'][i]) - float(X['world_landmark_0.y'][i]), 2) + pow(float(X['world_landmark_17.z'][i]) - float(X['world_landmark_0.z'][i]), 2))
            finger_tip_dist = math.sqrt(pow(float(X['world_landmark_20.x'][i]) - float(X['world_landmark_0.x'][i]), 2) + pow(float(X['world_landmark_20.y'][i]) - float(X['world_landmark_0.y'][i]), 2) + pow(float(X['world_landmark_20.z'][i]) - float(X['world_landmark_0.z'][i]), 2))
            th1 = 1.2
            th2 = 0.8
            if finger_tip_dist > th1 * ref_dist:
                finger_5.append('straight')
            elif finger_tip_dist < th1 * ref_dist and finger_tip_dist > th2 * ref_dist:
                finger_5.append('bent')
            else:
                finger_5.append('closed')

        th_high = 0.9
        th_low = 0.6
       

X = pd.DataFrame({'1': finger_1, '2': finger_2, '3': finger_3, '4': finger_4, '5': finger_5, 'letter': X['letter']})

X.to_csv('xd.csv')

X_train, X_test = train_test_split(X, test_size=0.2, random_state=24, stratify=X['letter'])
Y_train = X_train['letter']
Y_test = X_test['letter']
X_train = X_train.drop(columns=['letter'])
X_test = X_test.drop(columns=['letter'])

le = LabelEncoder()

X_train['1'] = le.fit_transform(X_train['1'])
X_train['2'] = le.fit_transform(X_train['2'])
X_train['3'] = le.fit_transform(X_train['3'])
X_train['4'] = le.fit_transform(X_train['4'])
X_train['5'] = le.fit_transform(X_train['5'])

X_test['1'] = le.fit_transform(X_test['1'])
X_test['2'] = le.fit_transform(X_test['2'])
X_test['3'] = le.fit_transform(X_test['3'])
X_test['4'] = le.fit_transform(X_test['4'])
X_test['5'] = le.fit_transform(X_test['5'])

model = RandomForestClassifier(n_estimators=100, random_state=24)

model.fit(X_train, Y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, pred)
print(accuracy)

confusion = confusion_matrix(Y_test, pred)
print('Confusion Matrix\n')
print(confusion)

y_unique = Y_test.unique()
print('\nClassification Report\n')
print(classification_report(Y_test, pred, target_names=y_unique))