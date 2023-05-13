import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.preprocessing import LabelEncoder

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

finger_1_orientation = []
finger_2_orientation = []
finger_3_orientation = []
finger_4_orientation = []
finger_5_orientation = []

for i in range(0, len(X)):
    for finger in fingers:
        # Obliczenie referencyjnej długości palca na podstawie budowy ludzkiej dłoni
        # oraz odległości końcowego punktu palca od początku palca
        if finger == '1':
            ref = math.sqrt(pow(float(X['world_landmark_1.x'][i]) - float(X['world_landmark_0.x'][i]), 2) + pow(float(X['world_landmark_1.y'][i]) - float(X['world_landmark_0.y'][i]), 2) + pow(float(X['world_landmark_1.z'][i]) - float(X['world_landmark_0.z'][i]), 2))
            finger_len = 2 * ref
            finger_tip_dist = math.sqrt(pow(float(X['world_landmark_4.x'][i]) - float(X['world_landmark_1.x'][i]), 2) + pow(float(X['world_landmark_4.y'][i]) - float(X['world_landmark_1.y'][i]), 2) + pow(float(X['world_landmark_4.z'][i]) - float(X['world_landmark_1.z'][i]), 2))
        elif finger == '2':
            ref = math.sqrt(pow(float(X['world_landmark_5.x'][i]) - float(X['world_landmark_0.x'][i]), 2) + pow(float(X['world_landmark_5.y'][i]) - float(X['world_landmark_0.y'][i]), 2) + pow(float(X['world_landmark_5.z'][i]) - float(X['world_landmark_0.z'][i]), 2))
            finger_len = 0.75 * ref
            finger_tip_dist = math.sqrt(pow(float(X['world_landmark_8.x'][i]) - float(X['world_landmark_5.x'][i]), 2) + pow(float(X['world_landmark_8.y'][i]) - float(X['world_landmark_5.y'][i]), 2) + pow(float(X['world_landmark_8.z'][i]) - float(X['world_landmark_5.z'][i]), 2))
        elif finger == '3':
            ref = math.sqrt(pow(float(X['world_landmark_9.x'][i]) - float(X['world_landmark_0.x'][i]), 2) + pow(float(X['world_landmark_9.y'][i]) - float(X['world_landmark_0.y'][i]), 2) + pow(float(X['world_landmark_9.z'][i]) - float(X['world_landmark_0.z'][i]), 2))
            finger_len = 0.75 * ref
            finger_tip_dist = math.sqrt(pow(float(X['world_landmark_12.x'][i]) - float(X['world_landmark_9.x'][i]), 2) + pow(float(X['world_landmark_12.y'][i]) - float(X['world_landmark_9.y'][i]), 2) + pow(float(X['world_landmark_12.z'][i]) - float(X['world_landmark_9.z'][i]), 2))
        elif finger == '4':
            ref = math.sqrt(pow(float(X['world_landmark_13.x'][i]) - float(X['world_landmark_0.x'][i]), 2) + pow(float(X['world_landmark_13.y'][i]) - float(X['world_landmark_0.y'][i]), 2) + pow(float(X['world_landmark_13.z'][i]) - float(X['world_landmark_0.z'][i]), 2))
            finger_len = 0.75 * ref
            finger_tip_dist = math.sqrt(pow(float(X['world_landmark_16.x'][i]) - float(X['world_landmark_13.x'][i]), 2) + pow(float(X['world_landmark_16.y'][i]) - float(X['world_landmark_13.y'][i]), 2) + pow(float(X['world_landmark_16.z'][i]) - float(X['world_landmark_13.z'][i]), 2))
        elif finger == '5':
            ref = math.sqrt(pow(float(X['world_landmark_17.x'][i]) - float(X['world_landmark_0.x'][i]), 2) + pow(float(X['world_landmark_17.y'][i]) - float(X['world_landmark_0.y'][i]), 2) + pow(float(X['world_landmark_17.z'][i]) - float(X['world_landmark_0.z'][i]), 2))
            finger_len = 0.75 * ref
            finger_tip_dist = math.sqrt(pow(float(X['world_landmark_20.x'][i]) - float(X['world_landmark_17.x'][i]), 2) + pow(float(X['world_landmark_20.y'][i]) - float(X['world_landmark_17.y'][i]), 2) + pow(float(X['world_landmark_20.z'][i]) - float(X['world_landmark_17.z'][i]), 2))
        
        th_high = 0.9
        th_low = 0.6


        if finger_tip_dist > th_high * finger_len:
            if finger == '1': finger_1.append('straight')
            if finger == '2': finger_2.append('straight')
            if finger == '3': finger_3.append('straight')
            if finger == '4': finger_4.append('straight')
            if finger == '5': finger_5.append('straight')
        elif th_low * finger_len < finger_tip_dist and th_high * finger_len > finger_tip_dist:
            if finger == '1': finger_1.append('bent')
            if finger == '2': finger_2.append('bent')
            if finger == '3': finger_3.append('bent')
            if finger == '4': finger_4.append('bent')
            if finger == '5': finger_5.append('bent')
        else:
            if finger == '1': finger_1.append('closed')
            if finger == '2': finger_2.append('closed')
            if finger == '3': finger_3.append('closed')
            if finger == '4': finger_4.append('closed')
            if finger == '5': finger_5.append('closed')

        

X = pd.DataFrame({'1': finger_1, '2': finger_2, '3': finger_3, '4': finger_4, '5': finger_5, 'letter': X['letter']})

X.to_csv('xd.csv')

X_train, X_test = train_test_split(X, test_size=0.1, random_state=24, stratify=X['letter'])
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