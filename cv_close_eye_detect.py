import pandas as pd
import numpy as np
import seaborn as sns
import keyboard
import matplotlib.pyplot as plt

import cv2
import time
import datetime


eye_cascPath = 'haarcascade_eye.xml'  #eye detect model
face_cascPath = 'haarcascade_frontalface_alt.xml'  #face detect model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+face_cascPath)
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades+eye_cascPath)

df=pd.DataFrame()
timenow=time.time()
eye_list=[]
row_dict=dict()
row_dict["time"]=0
row_dict["eye"]=False
row_dict["duration"]=0
cap = cv2.VideoCapture(1)
while 1:
    ret, img = cap.read()

    if ret:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        # print("Found {0} faces!".format(len(faces)))
        if len(faces) > 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
            frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
            eyes = eyeCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(10, 10),
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )
            row_dict["time"]=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if len(eyes) == 0:
                print('no eyes!!!')
                if row_dict["eye"]==True:   
                    row_dict["duration"]=time.time()-timenow
                    eye_list.append(row_dict.copy() )
                    # print(eye_list)
                    timenow=time.time()
                    row_dict["eye"]=False
            else:
                # print('eyes!!!')
                row_dict["eye"]=True
            if keyboard.is_pressed('='):  # if key '=' is pressed 
                print('You Pressed A Key!')
                df=pd.DataFrame.from_dict(eye_list,orient='columns')
                # print(eye_list)
                fig_plot=sns.lineplot(data=df, x="time", y="duration")
                fig_plot.set_xticklabels(fig_plot.get_xticklabels(),rotation = 90)
                
                fig = fig_plot.get_figure()
                fig.savefig("out.png") 
                df.to_csv("reading.csv")
                print("written to_csv")
                break  # finishing the loop
            # frame_tmp = cv2.resize(frame_tmp, (400, 400), interpolation=cv2.INTER_LINEAR)
            # cv2.imshow('Face Recognition', frame_tmp)
        waitkey = cv2.waitKey(1)
        if waitkey == ord('q') or waitkey == ord('Q'):
            cv2.destroyAllWindows()
            df=pd.DataFrame.from_dict(eye_list,orient='columns')
            # print(eye_list)
            fig_plot=sns.lineplot(data=df, x="time", y="duration")
            fig = fig_plot.get_figure()
            fig.savefig("out.png") 
            df.to_csv("reading.csv")
            print("written to_csv")
            break
        