from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import os
from ultralytics import YOLO
import numpy as np
from sort import *

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

model = YOLO("yolov8l.pt")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

vehicle_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", 
              "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli","carrot","hot dog","pizza",
              "donut","cake","chair","sofa","pottedplant","bed",
              "diningtable","toilet","tvmonitor","laptop","mouse",
              "remote","keyboard","cell phone","microwave","oven",
              "toaster","sink","refrigerator","book","clock",
              "vase","scissors","teddy bear","hair drier","toothbrush"]

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html", video_url="", logs="", vehicle_counts={})

@app.route("/upload", methods=["POST"])
def upload():
    if "video_file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["video_file"]
    confidence = float(request.form["confidence"])
    line_method = request.form["line_method"]
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(video_path)

    processed_video_path, logs, vehicle_counts = process_video(video_path, confidence)
    video_url = url_for("static", filename=os.path.basename(processed_video_path))
    return render_template("index.html", video_url=video_url, logs=logs, vehicle_counts=vehicle_counts)

def process_video(video_path, confidence):
    cap = cv2.VideoCapture(video_path)
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], "processed_video.mp4")
    vehicle_counts = {vehicle: 0 for vehicle in vehicle_classes}
    logs = []

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                currentClass = vehicle_classes[cls]
                if currentClass in vehicle_counts.keys() and conf > confidence:
                    detections = np.vstack([detections, [x1, y1, x2, y2, conf]])
                    vehicle_counts[vehicle_classes[cls]] += 1

        tracker.update(detections)
        out.write(frame)

    cap.release()
    out.release()
    return output_path, "Processing Complete!", vehicle_counts

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)
