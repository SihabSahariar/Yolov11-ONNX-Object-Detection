import cv2

from yolov11.yolocore import YOLODetector

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv11 object detector
model_path = 'models/yolov11n.onnx'
yolov11_detector =  YOLODetector(model_path= model_path , conf_thresh=0.5, iou_thresh=0.45)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids = yolov11_detector.detect(frame)
    combined_img = yolov11_detector.draw_detections(frame, boxes, scores, class_ids)
    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
