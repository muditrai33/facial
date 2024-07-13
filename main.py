from flask import Flask, request, jsonify
from flask_cors import CORS
from glob import glob
import os
import cv2
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

app = Flask(__name__)
CORS(app)

haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = "TrainingImageLabel/Trainner.yml"
trainimage_path = "TrainingImage"
studentdetail_path = "StudentDetails/studentdetails.csv"
attendance_path = "Attendance"

# Helper functions
def check_haarcascadefile():
    if not os.path.isfile(haarcasecade_path):
        raise FileNotFoundError(f'{haarcasecade_path} file is missing')

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def getImagesAndLabels(path):
    newdir = [os.path.join(path, f) for f in os.listdir(path)]
    imagePaths = [
        os.path.join(newdir[i], f)
        for i in range(len(newdir))
        for f in os.listdir(newdir[i])
    ]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids

def TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(haarcasecade_path)
    faces, Id = getImagesAndLabels(trainimage_path)
    recognizer.train(faces, np.array(Id))
    recognizer.save(trainimagelabel_path)
    res = "Image Trained successfully"  # +",".join(str(f) for f in Id)
    print(res)

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    Id = data['id']
    name = data['name']
    
    # Ensure directories exist
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    
    # Read or initialize serial number
    serial = 0
    if os.path.isfile("StudentDetails/studentdetails.csv"):
        with open("StudentDetails/studentdetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for _ in reader1:
                serial += 1
        serial = serial // 2
    else:
        with open("StudentDetails/studentdetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(['ID','NAME'])
        serial = 1

    # Capture images
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = cv2.CascadeClassifier(haarcasecade_path)
    sampleNum = 0
    os.makedirs(f"TrainingImage/{name}_{Id}")
    while True:
        ret, img = cam.read()
        if not ret:
            return jsonify({'message': 'Failed to grab frame'}), 500

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sampleNum += 1
            cv2.imwrite(f"TrainingImage/{name}_{Id}/{name}.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
            cv2.imshow('Taking Images', img)
        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 100:
            break
    cam.release()
    cv2.destroyAllWindows()
    
    # Save details
    with open("StudentDetails/studentdetails.csv", 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([Id, name])
    
    TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path)
    return jsonify({'message': 'Images Taken and Profile Saved'}), 200

@app.route('/attendance', methods=['POST'])
def attendance():
    now = time.time()
    future = now + 200
    data = request.get_json()
    subject = data.get('subject')

    if not subject:
        return jsonify({'message': 'Subject is required'}), 400

    check_haarcascadefile()
    assure_path_exists(f"{attendance_path}/{subject}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read(trainimagelabel_path)
    except:
        return jsonify({'message': 'Model not trained yet'}), 400
    
    faceCascade = cv2.CascadeClassifier(haarcasecade_path)
    if not os.path.isfile(studentdetail_path):
        return jsonify({'message': 'Student details not found'}), 400
    
    df = pd.read_csv(studentdetail_path)
    
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return jsonify({'message': 'Error: Camera not accessible'}), 500
    
    col_names = ["ID", "NAME"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    attendance = pd.DataFrame(columns=col_names)

    present_students = set()
    
    while True:
        ret, im = cam.read()
        if not ret:
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 70:
                present_students.add(Id)
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                name = df.loc[df["ID"] == Id]["NAME"].values[0]
                attendance.loc[len(attendance)] = [Id, name]
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 4)
                cv2.putText(im, f"{Id}-{name}", (x, y - 10), font, 1, (255, 255, 0), 2)
            else:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 4)
                cv2.putText(im, "Unknown", (x, y - 10), font, 1, (0, 0, 255), 2)
        
        # Break the loop after a certain time
        if time.time() > future:
            break

        cv2.imshow("Filling Attendance...", im)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    # Add the current date column and mark attendance
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    df[date] = df['ID'].apply(lambda x: 1 if x in present_students else 0)

    # Save unique attendance records
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    Hour, Minute, Second = timeStamp.split(":")
    fileName = f"{attendance_path}/{subject}/{subject}_{date}_{Hour}-{Minute}-{Second}.csv"
    df.to_csv(fileName, index=False, columns=["ID", "NAME", date])

    return jsonify(df.to_dict(orient='records')), 200


attendance_path='Attendance'
@app.route('/view-attendance', methods=['POST'])
def view_attendance():
    data = request.get_json()
    subject = data.get('subject')

    if not subject:
        return jsonify({'message': 'Please provide a subject name.'}), 400

    subject_path = os.path.join(attendance_path, subject)
    
    if not os.path.exists(subject_path):
        return jsonify({'message': f'The directory for subject "{subject}" does not exist.'}), 400

    filenames = glob(os.path.join(subject_path, f"{subject}*.csv"))
    if not filenames:
        return jsonify({'message': f'No CSV files found for subject "{subject}".'}), 400

    try:
        # Read student details
        student_df = pd.read_csv(studentdetail_path)

        # Initialize the attendance DataFrame with student details
        attendance_df = student_df[['ID', 'NAME']].copy()

        # Process each CSV file
        for filename in filenames:
            df = pd.read_csv(filename)
            date_str = os.path.basename(filename).split('_')[1]  # Assuming filename format includes date
            date_col = datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
            df = df[['ID', date_col]]
            attendance_df = attendance_df.merge(df, on='ID', how='left')

        # Fill NaN values with 0 for absent students
        attendance_df.fillna(0, inplace=True)
        attendance_df = attendance_df.astype({col: 'int32' for col in attendance_df.columns[2:]})

        # Calculate the attendance percentage
        total_days = len(attendance_df.columns) - 2  # Excluding ID and NAME columns
        attendance_df["Attendance"] = attendance_df.iloc[:, 2:].sum(axis=1) / total_days * 100
        attendance_df["Attendance"] = attendance_df["Attendance"].apply(lambda x: f'{x:.2f}%')

        # Save the consolidated attendance DataFrame
        attendance_df.to_csv(os.path.join(subject_path, "attendance.csv"), index=False)
    except Exception as e:
        print(str(e))
        return jsonify({'message': f'Error processing CSV files: {str(e)}'}), 500
    
    return jsonify(attendance_df.to_dict(orient='records')), 200


@app.route('/change-password', methods=['POST'])
def change_password():
    data = request.get_json()
    old_password = data['oldPassword']
    new_password = data['newPassword']
    
    # Replace this with your actual password change logic
    if old_password == 'your-old-password':  # Placeholder for actual password verification
        with open("password.txt", 'w') as f:
            f.write(new_password)
        return jsonify({'message': 'Password Changed Successfully'}), 200
    else:
        return jsonify({'message': 'Old Password is Incorrect'}), 400

if __name__ == '__main__':
    app.run(debug=True)
