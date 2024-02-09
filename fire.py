from ultralytics import YOLO
import cvzone
import cv2
import math
import datetime
model = YOLO('fire.pt')
classnames = ['fire']

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer


def main():
    cap = cv2.VideoCapture('0.mp4') 
    writer = create_video_writer(cap, "outp.mp4")
    while cap.isOpened():
        success,frame = cap.read()
        start = datetime.datetime.now()

        if success:
                #frame = cv2.resize(frame,(640,480))
                result = model(frame,stream=True)
                for info in result:
                    boxes = info.boxes
                    for box in boxes:
                            confidence = box.conf[0]
                            confidence = math.ceil(confidence * 100)
                            Class = int(box.cls[0])
                            if confidence > 50:
                                x1,y1,x2,y2 = box.xyxy[0]
                                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
                                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                        scale=1.5,thickness=2)
        end = datetime.datetime.now()
        total = (end - start).total_seconds()
        fps = f"FPS: {1 / total:.2f}"
        writer.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
               break               

if __name__=="__main__":
    main()