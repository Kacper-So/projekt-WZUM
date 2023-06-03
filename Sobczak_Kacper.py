import pickle
import pandas as pd
import math
import argparse
from pathlib import Path
import csv
from sklearn.preprocessing import LabelEncoder

# Funkcja define_properties przyjmuje dataframe z danymi wejściowymi
# Funkcja define_properties zwraca dataframe z cechami określonymi na podstawie danych wejściowych
# 
# Określane cechy :
# * 1-5 Na podstawie koordynatów landmarków określany jest stan poszczególnych palców ręki, dostępne są trzy stany: straight, bent, closed
# * finger_orientation_(1-5) Na podstawie koordynatów landmarków okeślany jest kierunek w którym wskazuje dany palec, dostępne są dwie orientacjie: up, down
# * hand_orientation Na podstawie koordynatów landmarków określana jest orientacja całej ręki na zdjęciu, dostępne są dwie orientacje: horizontal, vertical
#
def define_properties(X):
    fingers = ['1', '2', '3', '4', '5']
    finger_1 = []
    finger_2 = []
    finger_3 = []
    finger_4 = []
    finger_5 = []
    hand_orientation = []
    finger_2_orientation = []
    finger_3_orientation = []
    finger_4_orientation = []
    finger_5_orientation = []

    for i in range(0, len(X)):

        if float(X['landmark_0.y'][i]) < 1.2 * float(X['landmark_13.y'][i]) and float(X['landmark_0.y'][i]) > 0.8 * float(X['landmark_13.y'][i]):
            hand_orientation.append('horizontal')
        else:
            hand_orientation.append('vertical')


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
                    

                if float(X['landmark_5.y'][i]) < float(X['landmark_8.y'][i]):
                    finger_2_orientation.append('down')
                else:
                    finger_2_orientation.append('up')

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

                if float(X['landmark_9.y'][i]) < float(X['landmark_12.y'][i]):
                    finger_3_orientation.append('down')
                else:
                    finger_3_orientation.append('up')


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

                if float(X['landmark_13.y'][i]) < float(X['landmark_16.y'][i]):
                    finger_4_orientation.append('down')
                else:
                    finger_4_orientation.append('up')

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


                if float(X['landmark_17.y'][i]) < float(X['landmark_20.y'][i]):
                    finger_5_orientation.append('down')
                else:
                    finger_5_orientation.append('up')
        

    X = pd.DataFrame({'1': finger_1, '2': finger_2, '3': finger_3, '4': finger_4, '5': finger_5, 'finger_2_orientation': finger_2_orientation, 'finger_3_orientation': finger_3_orientation, 'finger_4_orientation': finger_4_orientation, 'finger_5_orientation': finger_5_orientation, 'hand_orientation': hand_orientation})
    return X

# przyjmowanie argumentów
parser = argparse.ArgumentParser()
parser.add_argument('test_data_file', type=str)
parser.add_argument('results_file', type=str)
args = parser.parse_args()
test_data_dir = Path(args.test_data_file)
results_file = Path(args.results_file)

# wczytanie danych testowych
test_dataset = []
with open(test_data_dir, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        test_dataset.append(row)

X_test = pd.DataFrame(test_dataset)
X_test = X_test.drop(columns=[''])
X_test = X_test.drop(columns=['handedness.label'])
X_test = X_test.drop(columns=['handedness.score'])

# określanie cech na podstawie danych wejściowych
X_test = define_properties(X_test)

le = LabelEncoder()
X_test['1'] = le.fit_transform(X_test['1'])
X_test['2'] = le.fit_transform(X_test['2'])
X_test['3'] = le.fit_transform(X_test['3'])
X_test['4'] = le.fit_transform(X_test['4'])
X_test['5'] = le.fit_transform(X_test['5'])
X_test['finger_2_orientation'] = le.fit_transform(X_test['finger_2_orientation'])
X_test['finger_3_orientation'] = le.fit_transform(X_test['finger_3_orientation'])
X_test['finger_4_orientation'] = le.fit_transform(X_test['finger_4_orientation'])
X_test['finger_5_orientation'] = le.fit_transform(X_test['finger_5_orientation'])
X_test['hand_orientation'] = le.fit_transform(X_test['hand_orientation'])

# wczytanie modelu
with open('model.pkl', 'rb') as plik:
    model = pickle.load(plik)

# wykonanie predykcji
pred = model.predict(X_test)

# zapis wyników
with results_file.open('wt') as output_file:
    for i in range(len(pred)):
        output_file.write(pred[i] + '\n')