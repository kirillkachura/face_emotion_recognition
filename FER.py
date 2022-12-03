

import numpy as np
import cv2
from tensorflow.keras import models
import mtcnn

# Инициалицация переменных и подключение к камере
video = cv2.VideoCapture(0)
ww = video.get(cv2.CAP_PROP_FRAME_WIDTH)
hh = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

print (ww, hh)

detector = mtcnn.MTCNN()

mod = models.load_model('model_1.h5')

font = cv2.FONT_HERSHEY_SIMPLEX
    
fontScale = 1
   
color = (255, 0, 0)
  
thickness = 2

emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']


# главный цикл анализа видеопотока
while (True):

    ret, frame = video.read()
        
    if ret:
	
        frame = cv2.resize(frame,(640, 480))

        faces = detector.detect_faces(frame)

        if faces:

            for i in faces:

                x, y, width, height = i['box']

                cv2.rectangle(frame, (x,y),(x+width,y+height), color = (255,0,0))

                if height >= width:
                    frame_cut = frame[y:y+height, int(x-(height-width)/2):int(x-(height-width)/2)+height]
                else:
                    frame_cut = frame[int(y-(width-height)/2):int(y-(width-height)/2)+width, x:x+width]

                if frame_cut.any():

                    frame_cut = cv2.cvtColor(frame_cut,cv2.COLOR_BGR2GRAY)

                    frame_cut = cv2.resize(frame_cut, (48,48))

                    frame_cut = frame_cut / 255

                    frame_to_mod = np.expand_dims(frame_cut, axis=0)

                    pred = mod.predict(frame_to_mod)

                    text = emotions[np.argmax(pred[0])]

                    cv2.putText(frame, text, (x, y-10), font, fontScale, color, thickness, cv2.LINE_AA)

                else:
                    pass

        cv2.imshow('frame',frame)



    inn = cv2.waitKey(1) & 0xFF	
    if inn == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

