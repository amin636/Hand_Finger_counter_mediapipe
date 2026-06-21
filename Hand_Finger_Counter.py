import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task")
except FileNotFoundError:
    print("not found hand_landmarker.task")
options=vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5
    )
detector=vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS=[
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

FINGER_TIPS=[8,12,16,20]
FINGER_JOINTS=[6,10,14,18]

frame_count=0

cap=cv2.VideoCapture(0)

if not cap.isOpened():
    print("camera not found!")
    exit()

while True :

    ret,frame=cap.read()
    if not ret:
        break

    frame=cv2.flip(frame,1)
    h,w,c=frame.shape

    mp_image=mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )
    frame_timestamp=int(frame_count*(1000/30))
    frame_count+=1
    detector_result=detector.detect_for_video(mp_image,frame_timestamp)
    finger_count=0
    if detector_result.hand_landmarks :
        for hand_landmark in detector_result.hand_landmarks:
            points=[]
            for landmark in hand_landmark:
                cx,cy=int(landmark.x * w),int(landmark.y * h)
                points.append((cx,cy))
                cv2.circle(frame,(cx,cy),5,(0,255,0),cv2.FILLED)
            for connect in HAND_CONNECTIONS:
                cv2.line(frame,points[connect[0]],points[connect[1]],(255,0,0),2)
            
            for i in range(4):
                nok=FINGER_TIPS[i]
                vasat=FINGER_JOINTS[i]
                if points[nok][1]<points[vasat][1]:
                    finger_count+=1
            if abs(points[4][0] - points[5][0]) > abs(points[2][0] - points[5][0]):
                finger_count += 1
                
        cv2.putText(
            frame, 
            f"finger : {finger_count}", 
            (50, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2, 
            (0, 0, 255), 
            3
        )
    cv2.imshow("Finger",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
detector.close()