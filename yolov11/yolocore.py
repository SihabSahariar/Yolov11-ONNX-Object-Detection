# Author: Sihab Sahariar
# Date: 2024-10-21
# License: MIT License
# Email: sihabsahariarcse@gmail.com

import argparse
import os
import sys
import os.path as osp
import cv2
import torch
import numpy as np
import onnxruntime as ort
from math import exp

# Constants and configurations
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']


COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

meshgrid = []
class_num = len(CLASSES)
headNum = 3
strides = [8, 16, 32]
mapSize = [[80, 80], [40, 40], [20, 20]]
input_imgH = 640
input_imgW = 640


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


class YOLODetector:
    def __init__(self, model_path='./yolov11n.onnx', conf_thresh=0.5, iou_thresh=0.45):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.ort_session = ort.InferenceSession(self.model_path)
        self.generate_meshgrid()

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def preprocess_image(img_src, resize_w, resize_h):
        image = cv2.resize(img_src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def generate_meshgrid(self):
        for index in range(headNum):
            for i in range(mapSize[index][0]):
                for j in range(mapSize[index][1]):
                    meshgrid.append(j + 0.5)
                    meshgrid.append(i + 0.5)

    def iou(self, xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
        xmin = max(xmin1, xmin2)
        ymin = max(ymin1, ymin2)
        xmax = min(xmax1, xmax2)
        ymax = min(ymax1, ymax2)

        innerWidth = max(0, xmax - xmin)
        innerHeight = max(0, ymax - ymin)

        innerArea = innerWidth * innerHeight
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        total = area1 + area2 - innerArea

        return innerArea / total

    def nms(self, detectResult):
        predBoxs = []
        sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

        for i in range(len(sort_detectboxs)):
            if sort_detectboxs[i].classId != -1:
                predBoxs.append(sort_detectboxs[i])
                for j in range(i + 1, len(sort_detectboxs), 1):
                    if sort_detectboxs[i].classId == sort_detectboxs[j].classId:
                        iou = self.iou(
                            sort_detectboxs[i].xmin, sort_detectboxs[i].ymin,
                            sort_detectboxs[i].xmax, sort_detectboxs[i].ymax,
                            sort_detectboxs[j].xmin, sort_detectboxs[j].ymin,
                            sort_detectboxs[j].xmax, sort_detectboxs[j].ymax
                        )
                        if iou > self.iou_thresh:
                            sort_detectboxs[j].classId = -1
        return predBoxs

    def postprocess(self, out, img_h, img_w):
        detectResult = []
        output = [out[i].reshape((-1)) for i in range(len(out))]
        scale_h = img_h / input_imgH
        scale_w = img_w / input_imgW

        gridIndex = -2

        for index in range(headNum):
            reg = output[index * 2 + 0]
            cls = output[index * 2 + 1]

            for h in range(mapSize[index][0]):
                for w in range(mapSize[index][1]):
                    gridIndex += 2

                    if class_num == 1:
                        cls_max = self.sigmoid(cls[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w])
                        cls_index = 0
                    else:
                        cls_max, cls_index = max(
                            [(self.sigmoid(cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]), cl)
                             for cl in range(class_num)]
                        )

                    if cls_max > self.conf_thresh:
                        regdfl = []
                        for lc in range(4):
                            locval = 0
                            sfsum = sum(exp(reg[((lc * 16) + df) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w])
                                        for df in range(16))
                            for df in range(16):
                                sfval = exp(reg[((lc * 16) + df) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) / sfsum
                                locval += sfval * df
                            regdfl.append(locval)

                        x1 = (meshgrid[gridIndex + 0] - regdfl[0]) * strides[index]
                        y1 = (meshgrid[gridIndex + 1] - regdfl[1]) * strides[index]
                        x2 = (meshgrid[gridIndex + 0] + regdfl[2]) * strides[index]
                        y2 = (meshgrid[gridIndex + 1] + regdfl[3]) * strides[index]

                        xmin = max(0, x1 * scale_w)
                        ymin = max(0, y1 * scale_h)
                        xmax = min(img_w, x2 * scale_w)
                        ymax = min(img_h, y2 * scale_h)

                        box = DetectBox(cls_index, cls_max, xmin, ymin, xmax, ymax)
                        detectResult.append(box)

        predBox = self.nms(detectResult)
        return predBox

    def detect(self, img_path):
        if isinstance(img_path, str):
            orig = cv2.imread(img_path)
        else:
            orig = img_path

        img_h, img_w = orig.shape[:2]
        image = self.preprocess_image(orig, input_imgW, input_imgH)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)

        pred_results = self.ort_session.run(None, {'data': image})
        predbox = self.postprocess(pred_results, img_h, img_w)

        boxes = []
        scores = []
        class_ids = []

        for box in predbox:
            boxes.append([int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)])
            scores.append(box.score)
            class_ids.append(box.classId)

        return boxes, scores, class_ids


    def draw_detections(self,image, boxes, scores, class_ids, mask_alpha=0.3):
        """
        Combines drawing masks, boxes, and text annotations on detected objects.
        
        Parameters:
        - image: Input image.
        - boxes: Array of bounding boxes.
        - scores: Confidence scores for each detected object.
        - class_ids: Detected object class IDs.
        - mask_alpha: Transparency of the mask overlay.
        """
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        mask_img = image.copy()

        # Draw bounding boxes, masks, and text annotations
        for class_id, box, score in zip(class_ids, boxes, scores):
            color = COLORS[class_id]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # Draw fill rectangle for mask
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            
            # Draw bounding box
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Prepare text (label and score)
            label = CLASSES[class_id]
            caption = f'{label} {int(score * 100)}%'
            
            # Calculate text size and position
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=font_size, thickness=text_thickness)
            th = int(th * 1.2)
            
            # Draw filled rectangle for text background
            cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
            
            # Draw text over the filled rectangle
            cv2.putText(det_img, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                        (255, 255, 255), text_thickness, cv2.LINE_AA)

        # Blend the mask image with the original image
        det_img = cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

        return det_img
