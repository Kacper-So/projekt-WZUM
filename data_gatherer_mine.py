from typing import Union, NamedTuple

import tkinter.messagebox
from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import pandas as pd
import cv2
import mediapipe as mp
import os
import csv

folder_path = './data'
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if any(f.endswith(ext) for ext in image_extensions)]
images_dict = {}
for file in image_files:
    img = cv2.imread(file)
    if img is not None:
        images_dict[os.path.basename(file)] = img


class_dict = {}
with open('.\data\_annotations.csv') as plik:
    czytnik = csv.reader(plik)
    for wiersz in czytnik:
        class_dict[wiersz[0]] = wiersz[3]

# MediaPipe config
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

filename: Union[None, str] = None
df: Union[None, pd.DataFrame] = None

file = fd.asksaveasfile(
    initialfile='dataset.csv',
    defaultextension='.csv',
    filetypes=[('All Files', '*.*'), ('csv files', '*.csv')], mode='w'
)
filename = file.name
file.close()
columns = ['landmark_'+str(i)+'.'+a for i in range(21) for a in ['x', 'y', 'z']]
columns += ['world_landmark_' + str(i) + '.' + a for i in range(21) for a in ['x', 'y', 'z']]
columns += ['handedness', 'letter']
df = pd.DataFrame(columns=columns)
df.to_csv(filename)

for image in images_dict:
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1
    ) as hands:
        def show_frames():
            global results
            cv2image = images_dict[image]
            cv2image.flags.writeable = False
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            results = hands.process(cv2image)

            cv2image.flags.writeable = True
            if results.multi_hand_landmarks:
                # for hand_landmarks in results.multi_hand_landmarks:
                #     mp_drawing.draw_landmarks(
                #         cv2image,
                #         hand_landmarks,
                #         mp_hands.HAND_CONNECTIONS,
                #         mp_drawing_styles.get_default_hand_landmarks_style(),
                #         mp_drawing_styles.get_default_hand_connections_style()
                #     )

                landmarks = []
                for ld in results.multi_hand_landmarks[0].landmark:
                    landmarks += [ld.x, ld.y, ld.z]
                for ld in results.multi_hand_landmarks[0].landmark:
                    landmarks += [ld.x, ld.y, ld.z]
                landmarks += [results.multi_handedness[0].classification[0].score]
                landmarks += [class_dict[image]]
                df.loc[len(df)] = landmarks
                df.to_csv('dataset.csv')

        show_frames()