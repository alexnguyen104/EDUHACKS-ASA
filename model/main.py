from ultralytics import YOLO
import cv2
import math
from classess import classNames
import pyttsx3
import threading

# use our testing video
#source = "videos/testing.mp4"

# use camera
source = 0
model = YOLO('yolov8n.pt')
x = classNames
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(source)
engine = pyttsx3.init() # object creation
state = "ok"

# Video writer
video_writer = cv2.VideoWriter("recorded_video.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       int(cap.get(5)),
                       (int(cap.get(3)), int(cap.get(4))))

def say(state):
    engine.say(f"{state}")
    engine.runAndWait()

while cap.isOpened:
    width  = cap.get(3)  # float `width`
    height = cap.get(4)  # float `height`

    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    result = model(im0, stream=True, device=0) #remove device=0 to use CPU if GPU is not available on your computer
    for r in result:
        boxes = r.boxes

        #region
        reg_w = 250
        x_reg1 = int(width/2 - reg_w/2)
        y_reg1 = 50
        x_reg2 = int(width/2 + reg_w/2)
        y_reg2 = int(height - 50)
        cv2.rectangle(im0,(x_reg1, y_reg1),(x_reg2, y_reg2),(0,255,0),2)


        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w = x2 - x1
            h = y2 - y1
            
            state = "ok"

            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])

            check_invasion = (x_reg2 - x1 >= 0 and x_reg2 - x1 < reg_w) or (x2 - x_reg1 >= 0 and x2 - x_reg1 < reg_w) or (x_reg2 - x1 > reg_w and x2 - x_reg1 > reg_w)
            check_size = w*h > 50000 #estimated size to get expected distance -> not to accurate but it is acceptable

            if(check_invasion and check_size):
                cv2.rectangle(im0,(x1,y1),(x2,y2),(0,0,255),3)
                cv2.putText(im0,f'{x[cls]}{conf}',(x1,y1),font,1,(0,255,0))
                if x_reg2 - x1 <= w/2:
                    state = "move left"
                elif x2 - x_reg1 <= w/2:
                    state = "move right"
                else:
                    state = "stop"

                threading.Thread(target=say, args=(state, )).start()
                cv2.putText(im0,state,(x_reg1,y_reg1),font,1,(0,255,0),3)       
                cv2.putText(im0,x[cls],(x1,y1),font,1,(0,255,0),3) # add label for detected object -> can be used in telling blind people who is coming

            else:
                cv2.rectangle(im0,(x1,y1),(x2,y2),(255,0,0),3)

        video_writer.write(im0)
                        
    cv2.imshow('me',im0)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()


