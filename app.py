import numpy as np
import pandas as pd
import cv2
import cvzone
import math
from sort import *  # Ensure that the Sort library is installed and in your environment
import streamlit as st
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8l.pt")


cumulative_time = {
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorbike": 0
}

vehicle_times = {
    "car": 5,
    "truck": 8,
    "bus": 10,
    "motorbike": 3
}

total_Time = 0

traffic_light_status = "green"  # Current traffic light state

# Set class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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

st.title("Traffic Detection App")
st.text("Vehicle detection and tracking with YOLO")

st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Level Threshold:", 0.0, 1.0, 0.3)

uploaded_file = st.file_uploader("Upload a Video File (MP4)", type=["mp4"])




if uploaded_file is not None:
    temp_video_path = f"temp_video.mp4"
    with open(temp_video_path, 'wb') as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_video_path)  # Use uploaded video file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Load mask and graphics
    mask = cv2.imread("mask.png")  # Mask to filter the region of interest
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    

    # Streamlit image holder for frames
    frame_window = st.image([])
    counter_text= st.empty()
    log_text= st.empty()
    log_time= st.empty()
    table_text = st.empty()

    vehicle_counts = {
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorbike": 0,
    "person": 0
    }

    method_choice = st.sidebar.radio("Select Method to Set Scanning Line Position", ("Slider", "Text Input"))

    default_x1, default_y1, default_x2, default_y2 = int(frame_width * 0.3), int(frame_height * 0.6), int(frame_width * 0.8),  int(frame_height * 0.6)

    if method_choice== "Slider":
        x1 = st.sidebar.slider("Line Start X", 0, frame_width, default_x1)
        y1 = st.sidebar.slider("Line Start Y", 0, frame_height, default_y1)
        x2 = st.sidebar.slider("Line End X", 0, frame_width, default_x2)
        y2 = st.sidebar.slider("Line End Y", 0, frame_height, default_y2)

        input_x1, input_y1, input_x2, input_y2 = x1, y1, x2, y2

    elif method_choice=="Text Input":    
        text_x1 = st.sidebar.text_input("Start X Position (Text Input)", value=str(default_x1))
        text_y1 = st.sidebar.text_input("Start Y Position (Text Input)", value=str(default_y1))
        text_x2 = st.sidebar.text_input("End X Position (Text Input)", value=str(default_x2))
        text_y2 = st.sidebar.text_input("End Y Position (Text Input)", value=str(default_y2))

        try:
            line_x1 = int(text_x1) if text_x1.isdigit() and 0 <= int(text_x1) <= frame_width else default_x1
            line_y1 = int(text_y1) if text_y1.isdigit() and 0 <= int(text_y1) <= frame_height else default_y1
            line_x2 = int(text_x2) if text_x2.isdigit() and 0 <= int(text_x2) <= frame_width else default_x2
            line_y2 = int(text_y2) if text_y2.isdigit() and 0 <= int(text_y2) <= frame_height else default_y2
    
        except ValueError:
            line_x1, line_y1, line_x2, line_y2 = default_x1, default_y1, default_x2, default_y2
        
        input_x1, input_y1, input_x2, input_y2 = line_x1, line_y1, line_x2, line_y2    

    limits = [input_x1, input_y1, input_x2, input_y2]
    #limits = [int(frame_width * 0.3), int(frame_height * 0.6), int(frame_width * 0.8), int(frame_height * 0.6)]
    totalCount = []
    # Resize the mask to match the video frame dimensions
    mask = cv2.resize(mask, (frame_width, frame_height))
    # Process each frame and display it
    while cap.isOpened():
        ret, img = cap.read()
        if not ret or traffic_light_status == "red":
            break
        img = cv2.GaussianBlur(img, (5, 5), 0)
        imgRegion = cv2.bitwise_and(img, mask)
        results = model(imgRegion, stream=True)

        #imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
        #img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
        
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence and class
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                # Filter vehicles based on confidence threshold set by user
                #if currentClass in ["car", "truck", "bus", "motorbike"] and conf > confidence_threshold:
                    #currentArray = np.array([x1, y1, x2, y2, conf])
                    #detections = np.vstack((detections, currentArray))
                if currentClass in vehicle_counts.keys() and conf > confidence_threshold:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        # Update the tracker with the current detections
        resultsTracker = tracker.update(detections)

        # Draw counting line and tracked objects
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            # Display detection and ID
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 255, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0,x1), max(35,y1)),
                               scale=1, thickness=2, colorR=(0, 0, 0))

            # Count vehicle if it crosses the line
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    
                    
                    color = (0, 255, 0)  # Default to green for cars
                    if currentClass == "truck":
                        color = (255, 0, 0)  # Red for trucks
                    elif currentClass == "motorbike":
                        color = (0, 255, 255)  # Cyan for motorbikes
                    elif currentClass == "bus":
                        color = (255, 255, 0)  # Yelow for buses
                    elif currentClass == "person":
                        color = (255, 0, 255)  # Magenta for persons
                    
                    if currentClass not in vehicle_counts:
                        vehicle_counts[currentClass] =0
                    
                    if currentClass in vehicle_times:
                        total_Time += vehicle_times[currentClass]
                    
                    vehicle_counts[currentClass] += 1
                    cv2.line(img,(limits[0], limits[1]), (limits[2], limits[3]), color ,5) 

                    if currentClass not in vehicle_counts:
                        vehicle_counts[currentClass]=0
                    vehicle_counts[currentClass]+= 1

                    #print(f"Total Time: {total_Time}")  
                    #st.text(f"Debug: Total Time = {total_Time}")

    
    
        # Display traffic light status
        if total_Time >= 40 and total_Time < 45:
            traffic_light_status = "yellow"
            cv2.circle(img, (50, 50), 20, (0, 255, 255), -1)  # Yellow circle
        elif total_Time >= 45:
            traffic_light_status = "red"
            cv2.circle(img, (50, 50), 20, (0, 0, 255), -1)  # Red circle
            #cap.release()
            #break
        else:
            traffic_light_status = "green"
            cv2.circle(img, (50, 50), 20, (0, 255, 0), -1)  # Green circle
        
        
        cv2.putText(img, f"Total Time: {total_Time}s", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Update vehicle count and logs
        vehicle_count = len(totalCount)
        #counter_text.markdown(f"**Total Vehicles Passed:** {vehicle_count}")
        filtered_vehicle_counts = {k: v for k, v in vehicle_counts.items() if k != "person"}

        vehicle_df = pd.DataFrame.from_dict(filtered_vehicle_counts, orient="index", columns=["Count"])
        table_text.table(vehicle_df)
         # Display logs
        log_text.markdown(f"### Logs\n- **Vehicles Passed:** {vehicle_count}\n- **Last Detected Vehicle ID:** {id if 'id' in locals() else 'N/A'}")   
        
        log_time.markdown(f"""
        - **Total Time Accumulated:** {total_Time}s
        """)
        # Show total count on the frame
        #cv2.putText(img,str(len(totalCount)),(255 ,100),cv2.FONT_HERSHEY_PLAIN ,5,(50 ,50 ,255),8)

        # Update Streamlit frame window with processed image
        frame_window.image(img ,channels="BGR")

    cap.release()
    if traffic_light_status== "red":
        st.warning("traffic is at halt")
else:
    st.warning("Please upload a video file to start processing.")

