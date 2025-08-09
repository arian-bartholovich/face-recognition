import cv2
import os
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def prepare_training_data(data_folder_path):
    faces = []
    labels = []
    label_map = {}
    label_count = 0

    for person_name in os.listdir(data_folder_path):
        person_path = os.path.join(data_folder_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[label_count] = person_name

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            # Read the image in grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Unable to read {image_path}")
                continue
            faces.append(image)
            labels.append(label_count)

        label_count += 1

    return faces, labels, label_map

def train_recognizer(faces, labels):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    return face_recognizer

def recognize_faces(test_image_path, face_recognizer, label_map):
    test_img = cv2.imread(test_image_path)
    if test_img is None:
        print(f"Error: Could not read {test_image_path}")
        return

    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces_rects:
        roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(roi)
        print(f"Recognized: {label_map[label]} with confidence {confidence}")

        # Draw rectangle around face
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Write label and confidence
        cv2.putText(test_img, f"{label_map[label]} ({int(confidence)})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    data_folder = "faces_dataset"  # Folder containing subfolders of person images

    print("Preparing training data...")
    faces, labels, label_map = prepare_training_data(data_folder)

    print("Training recognizer...")
    face_recognizer = train_recognizer(faces, labels)

    # Open a file dialog to select the test image
    print("Please select a test image file...")
    Tk().withdraw()  # Hide the root Tk window
    test_image = askopenfilename(title="Select Test Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    if test_image:
        print(f"Running face recognition on: {test_image}")
        recognize_faces(test_image, face_recognizer, label_map)
    else:
        print("No file selected. Exiting.")
