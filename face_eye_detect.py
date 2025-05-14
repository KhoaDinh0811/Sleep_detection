import cv2 as cv
import numpy as np
import time
from tensorflow.keras.models import load_model
from pygame import mixer 

mixer.init()
sound = mixer.Sound('alarm.wav') 

model = load_model('eye_state_model.h5')  
url = "http://192.168.1.20:4747/video"
cam = cv.VideoCapture(url)
face = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
leye = cv.CascadeClassifier("haarcascade_lefteye_2splits.xml")
reye = cv.CascadeClassifier("haarcascade_righteye_2splits.xml")

EYE_CLOSED_THRESHOLD = 0.5  
DROWSY_TIME_THRESHOLD = 2.0  

eye_closed_start_time = None
drowsy_alert_active = False

def predict_eye_state(eye_img):
    if eye_img is None or eye_img.size == 0:
        return 1  
    eye_img = cv.cvtColor(eye_img, cv.COLOR_BGR2GRAY)
    eye_img = cv.resize(eye_img, (24, 24))  
    eye_img = eye_img / 255.0 
    eye_img = eye_img.reshape(1, 24, 24, 1) 
    
    pred = model.predict(eye_img)[0][0] 
    return pred

while True:
    ret, frame = cam.read()
    if not ret:
        break 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
    
    # Nếu không phát hiện khuôn mặt, tiếp tục vòng lặp
    if len(faces_rect) == 0:
        cv.putText(frame, "No face detected", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if drowsy_alert_active:
            sound.stop()
            drowsy_alert_active = False
        eye_closed_start_time = None
    
    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Phát hiện mắt trái
        left_eye_closed = True
        left_eyes = leye.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors = 5)
        for (ex, ey, ew, eh) in left_eyes:

            left_eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            left_eye_status = predict_eye_state(left_eye_roi)
            eye_state_text = f"Left: {left_eye_status:.2f}"
            
            color = (0, 255, 0) if left_eye_status >= EYE_CLOSED_THRESHOLD else (0, 0, 255)
            cv.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, thickness=2)
            cv.putText(frame, eye_state_text, (x+ex, y+ey-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if left_eye_status >= EYE_CLOSED_THRESHOLD:
                left_eye_closed = False
        
        # Phát hiện mắt phải
        right_eye_closed = True
        right_eyes = reye.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors = 5)
        for (ex, ey, ew, eh) in right_eyes:

            right_eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            right_eye_status = predict_eye_state(right_eye_roi)
            eye_state_text = f"Right: {right_eye_status:.2f}"
            
            color = (0, 255, 0) if right_eye_status >= EYE_CLOSED_THRESHOLD else (0, 0, 255)
            cv.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, thickness=2)
            cv.putText(frame, eye_state_text, (x+ex, y+ey-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if right_eye_status >= EYE_CLOSED_THRESHOLD:
                right_eye_closed = False
        
        both_eyes_closed = left_eye_closed and right_eye_closed
        
        if both_eyes_closed:
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            
            closed_duration = time.time() - eye_closed_start_time
            
            # Kiểm tra nếu mắt đóng quá lâu
            if closed_duration >= DROWSY_TIME_THRESHOLD:
                cv.putText(frame, "DROWSINESS ALERT!", (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                if not drowsy_alert_active:
                    sound.play(-1)  # -1 để phát lặp lại
                    drowsy_alert_active = True
            
            # Hiển thị thời gian đóng mắt
            cv.putText(frame, f"Eyes closed: {closed_duration:.1f}s", (x, y+h+20), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # Nếu mắt mở lại
            if eye_closed_start_time is not None:
                eye_closed_start_time = None
                
            # Dừng cảnh báo nếu đang kích hoạt
            if drowsy_alert_active:
                sound.stop()
                drowsy_alert_active = False
            
            cv.putText(frame, "Eyes open", (x, y+h+20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv.imshow("Drowsiness Detection", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv.destroyAllWindows()
if drowsy_alert_active:
    sound.stop()