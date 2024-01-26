# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 23:18:00 2023

@author: meissa
"""

import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()

# Check if the returned value is a single scalar index or a list of indices
if isinstance(unconnected_layers, int):
    unconnected_layers = [unconnected_layers]

output_layers = [layer_names[i - 1] for i in unconnected_layers]


# Start capturing video from the default camera (change index if needed)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Convert frame to blob for YOLO input
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class ID 0 is for people in COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Count people and draw bounding boxes
    font = cv2.FONT_HERSHEY_PLAIN
    count = 0
    for i in range(len(boxes)):
        if i in indexes:
            count += 1
            x, y, w, h = boxes[i]
            label = f"Person {count}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), font, 1, (0, 255, 0), 2)

    # Display count and video feed with bounding boxes
    cv2.putText(frame, f"Number of people: {count}", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.imshow("Real-time People Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
