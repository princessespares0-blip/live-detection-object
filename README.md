🎥 Live Object Detection & Tracking
📌 Project Overview

Live Object Detection & Tracking is a real-time Artificial Intelligence web application built using Streamlit, YOLOv8, OpenCV, and WebRTC.

The system allows users to access their webcam through a browser and detect live objects instantly. It identifies objects such as people, bottles, and cell phones, then displays bounding boxes, confidence scores, object counts, FPS (Frames Per Second), and alert notifications.

This project demonstrates how computer vision and machine learning can be integrated into an interactive web application for real-time object monitoring.

🚀 Features
Real-time webcam object detection
Live object tracking with bounding boxes
Adjustable confidence threshold
FPS (Frames Per Second) monitoring
Object counting
Alert system for selected objects
Interactive browser-based interface
Clean and modern UI design
🛠 Technologies Used
Python
Streamlit
Streamlit WebRTC
YOLOv8 (Ultralytics)
OpenCV
AV
CSS Styling
📂 Project Structure
live-object-detection/
│── app.py
│── requirements.txt
│── packages.txt
│── runtime.txt
│── yolov8s.pt
│── README.md
⚙ Installation
1. Clone the repository
git clone https://github.com/your-username/live-object-detection.git
cd live-object-detection
2. Install dependencies
pip install -r requirements.txt
3. Run the application
streamlit run app.py
📦 Requirements

Install the required packages:

streamlit
streamlit-webrtc
ultralytics
opencv-python-headless
av
torch
torchvision
numpy
🎮 How to Use
Launch the app
Allow webcam access
Adjust the confidence threshold using the sidebar
Show objects to the camera
View:
Detected object labels
Confidence scores
FPS
Object counts
Alerts
🔍 How It Works
Model Loading

The application loads the YOLOv8 Small Model (yolov8s.pt) for object detection.

model = YOLO("yolov8s.pt")
Video Processing

Each webcam frame is processed in real time:

Captures webcam input
Runs object detection
Draws bounding boxes
Displays labels
Counts detected objects
Shows FPS
Alerts

The system displays alerts when these objects are detected:

Person
Cell Phone
Bottle
📊 Performance Evaluation
Accuracy Under Different Lighting Conditions

The system performs best under bright and normal indoor lighting.

Bright lighting: High accuracy
Normal indoor lighting: Stable detection
Dim lighting: Slight reduction in accuracy
Low lighting: Detection becomes less reliable
Performance Observation

Based on observation, the system runs smoothly on mid-range devices and provides near real-time detection. Slight lag may occur when multiple objects appear at the same time or when objects move too quickly.

💡 Applications

This project can be used for:

Smart surveillance
Classroom demonstrations
Object monitoring systems
AI learning projects
Computer vision research
🔮 Future Improvements

Possible enhancements:

Object tracking history
Screenshot capture
Detection recording
Custom trained models
Mobile optimization
👨‍💻 Author

Developed as a real-time AI computer vision project using Streamlit and YOLOv8.

📜 License

This project is for educational and learning purposes.