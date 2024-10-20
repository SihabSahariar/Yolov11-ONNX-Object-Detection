import cv2

from yolov11.yolocore import YOLODetector

# Initialize YOLOv11 object detector
model_path = 'models/yolov11n.onnx'
yolov11_detector =  YOLODetector(model_path= model_path , conf_thresh=0.1, iou_thresh=0.45)

# Load image
image_path = 'test2.jpg'
image = cv2.imread(image_path)

# Detect objects in the image
boxes, scores, class_ids = yolov11_detector.detect(image)
frame = yolov11_detector.draw_detections(image, boxes, scores, class_ids)

# Save the image
cv2.imwrite("detected_objects_2.jpg", frame)

# Display the image
cv2.imshow("Detected Objects", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
