from flask import Flask, render_template, Response
import cv2
import mtcnn
from tensorflow.keras import models
import numpy as np


app = Flask(__name__)

detector = mtcnn.MTCNN()

mod = models.load_model('model_2.h5')

font = cv2.FONT_HERSHEY_SIMPLEX
    
fontScale = 1
   
color = (255, 0, 0)
  
thickness = 2

emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
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

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        
            


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
