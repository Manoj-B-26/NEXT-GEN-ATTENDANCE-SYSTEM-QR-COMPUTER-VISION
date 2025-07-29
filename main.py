from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
import cv2
import os
import numpy as np
import csv
from datetime import datetime
import qrcode
from pyzbar.pyzbar import decode
import io
import base64
from PIL import Image

app = Flask(__name__)
app.secret_key = "qr_attendance_system_secret_key"

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
ATTENDANCE_LOG_PATH = os.path.join(BASE_DIR, 'attendance_log.csv')
QRCODES_PATH = os.path.join(BASE_DIR, 'qrcodes')
RECOGNIZED_FACES_PATH = os.path.join(BASE_DIR, 'recognized_faces')
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# Create necessary directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(QRCODES_PATH, exist_ok=True)
os.makedirs(RECOGNIZED_FACES_PATH, exist_ok=True)

# Initialize the face cascade classifier
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Initialize the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Global variables
camera = None
qr_data = None
attendance_marked = False

# Function to load the dataset and train the recognizer
def train_recognizer():
    faces = []
    labels = []
    label_dict = {}  # Dictionary to map labels to names
    
    # Loop through the dataset folder and collect images and labels
    for label_folder in os.listdir(DATASET_PATH):
        label_folder_path = os.path.join(DATASET_PATH, label_folder)
        
        if os.path.isdir(label_folder_path):
            try:
                label = int(label_folder)  # Assuming the folder name is the label
                label_dict[label] = label_folder  # Map label to name
                
                # Collect images in each folder
                for image_name in os.listdir(label_folder_path):
                    image_path = os.path.join(label_folder_path, image_name)
                    
                    # Load the image, convert to grayscale, and append to faces list
                    img = cv2.imread(image_path)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces.append(gray)
                        labels.append(label)
            except ValueError:
                # Skip folders that cannot be converted to int
                continue
    
    # Train the recognizer if we have data
    if len(faces) > 0 and len(labels) > 0:
        recognizer.train(faces, np.array(labels))
        return label_dict
    return {}

# Function to log attendance in CSV
def log_attendance(name, confidence, filename):
    # Create CSV file if it doesn't exist
    if not os.path.isfile(ATTENDANCE_LOG_PATH):
        with open(ATTENDANCE_LOG_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Confidence", "Image Filename", "Timestamp"])
            
    with open(ATTENDANCE_LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the name, confidence score, image filename, and current time (attendance mark)
        writer.writerow([name, confidence, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# Function to generate frames for video streaming
def generate_frames(mode='capture', person_id=None):
    global camera, qr_data, attendance_marked
    
    # Initialize camera if not already done
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    count = 0
    label_dict = train_recognizer()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if mode == 'capture' and person_id:
            # Capture mode: Detect and save face images
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Save the face image if count is less than 30
                if count < 30:
                    person_folder = os.path.join(DATASET_PATH, person_id)
                    os.makedirs(person_folder, exist_ok=True)
                    face_img = gray[y:y + h, x:x + w]
                    face_filename = os.path.join(person_folder, f"img{count + 1}.jpg")
                    cv2.imwrite(face_filename, face_img)
                    count += 1
                    cv2.putText(frame, f"Captured: {count}/30", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        elif mode == 'scan_qr':
            # QR code scanning mode
            qr_codes = decode(frame)
            for qr_code in qr_codes:
                qr_data = qr_code.data.decode('utf-8')
                cv2.putText(frame, f"QR: {qr_data}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        elif mode == 'recognize':
            # Face recognition mode
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                
                # Region of interest for face recognition
                roi_gray = gray[y:y + h, x:x + w]
                
                try:
                    # Only predict if the recognizer has been trained
                    if label_dict:
                        label, confidence = recognizer.predict(roi_gray)
                        
                        # Get the recognized name from the label dictionary
                        name = label_dict.get(label, "Unknown")
                        text = f"Name: {name}, Confidence: {confidence:.2f}"
                        
                        # Display details on the screen
                        cv2.putText(frame, text, (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # If the person is recognized and attendance is not marked yet
                        if name != "Unknown" and not attendance_marked and qr_data:
                            # Save the image of the recognized face
                            recognized_face_filename = os.path.join(
                                RECOGNIZED_FACES_PATH, 
                                f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            )
                            cv2.imwrite(recognized_face_filename, roi_gray)
                            
                            # Log the attendance with additional details
                            log_attendance(name, confidence, recognized_face_filename)
                            
                            # Mark the attendance as done
                            attendance_marked = True
                            cv2.putText(frame, "Attendance Marked!", (10, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except:
                    cv2.putText(frame, "Recognition Error", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame to the streaming response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Exit capture mode after 30 images
        if mode == 'capture' and count >= 30:
            break

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_faces', methods=['GET', 'POST'])
def capture_faces():
    global camera
    
    if request.method == 'POST':
        person_id = request.form['person_id']
        if person_id:
            return render_template('capture.html', person_id=person_id)
        else:
            flash('Please enter a valid ID', 'error')
    
    return render_template('capture_form.html')

@app.route('/video_feed/<mode>/<person_id>')
def video_feed(mode, person_id):
    return Response(generate_frames(mode=mode, person_id=person_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<mode>')
def video_feed_mode(mode):
    return Response(generate_frames(mode=mode),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_qr', methods=['GET', 'POST'])
def generate_qr():
    if request.method == 'POST':
        room_no = request.form['room_no']
        period_no = request.form['period_no']
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        qr_data = f"Room: {room_no}, Period: {period_no}, Timestamp: {timestamp}"
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_data)
        qr.make(fit=True)
        
        img = qr.make_image(fill="black", back_color="white")
        
        file_path = os.path.join(QRCODES_PATH, f"QR_Room{room_no}_Period{period_no}.png")
        img.save(file_path)
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return render_template('qr_result.html', 
                               qr_image=img_str, 
                               room_no=room_no, 
                               period_no=period_no,
                               file_path=file_path)
    
    return render_template('generate_qr.html')

@app.route('/scan_qr')
def scan_qr():
    global qr_data, attendance_marked
    # Reset the global variables
    qr_data = None
    attendance_marked = False
    return render_template('scan_qr.html')

@app.route('/get_qr_data')
def get_qr_data():
    global qr_data
    return jsonify({'qr_data': qr_data})

@app.route('/face_recognition')
def face_recognition():
    global qr_data, attendance_marked
    if qr_data:
        attendance_marked = False
        return render_template('recognition.html', qr_data=qr_data)
    else:
        flash('Please scan a QR code first', 'error')
        return redirect(url_for('scan_qr'))

@app.route('/view_attendance')
def view_attendance():
    attendance_data = []
    if os.path.isfile(ATTENDANCE_LOG_PATH):
        with open(ATTENDANCE_LOG_PATH, mode='r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Skip header
            for row in csv_reader:
                attendance_data.append(row)
    
    return render_template('attendance.html', attendance_data=attendance_data)

@app.route('/train_model')
def train_model():
    label_dict = train_recognizer()
    if label_dict:
        flash(f'Model trained successfully with {len(label_dict)} users', 'success')
    else:
        flash('No training data available', 'error')
    return redirect(url_for('index'))

@app.route('/shutdown_camera')
def shutdown_camera():
    global camera
    if camera:
        camera.release()
        camera = None
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)