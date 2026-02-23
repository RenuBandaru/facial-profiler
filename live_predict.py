# Importing necessary libraries
import cv2
import tensorflow as tf
import numpy as np
from mtcnn import MTCNN
import os

# Load the trained model
model = tf.keras.models.load_model("facial_multitask_model.keras")

# Label mappings
age_labels = ['0-9', '10-19', '10-19', '20-29', '30-39', '40-49', '50-59', '60+']
gender_labels = {0: 'Male', 1: 'Female'}
emotion_labels = {1: 'Surprise', 2: 'Sad', 3: 'Disgust', 4: 'Happy', 5: 'Fear', 6: 'Angry', 7: 'Neutral'}

# Preprocessing function for the input image
def load_and_preprocess_image(image):
    
    img = cv2.resize(image, (224, 224))
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    
    return img

# Initialize MTCNN for face detection
detector = MTCNN()

# Start video capture
cap = cv2.VideoCapture(0) #Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces in the frame
    faces = detector.detect_faces(frame)
    
    for face in faces:
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)  # Ensure coordinates are non-negative
        face_img = frame[y:y+height, x:x+width]
        
        # Preprocess the face image
        preprocessed_face = load_and_preprocess_image(face_img)[np.newaxis, ...]  # Add batch dimension
        age_logits, gender_logits, emotion_logits = model(preprocessed_face)
        
        # Extract predictions for each task
        age_pred = np.argmax(age_logits, axis=-1)[0]
        gender_pred = np.argmax(gender_logits, axis=-1)[0]
        emotion_pred = np.argmax(emotion_logits, axis=-1)[0]
        
        # Draw results on the frame
        label = f"Age:{age_labels[age_pred]} G:{gender_labels[gender_pred]} E:{emotion_labels[emotion_pred]}"
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Make predictions
        # preprocessed_face = load_and_preprocess_image(face_img)
        # predictions = model(np.expand_dims(preprocessed_face, axis=0))  # Add batch dimension
        # age_logits, gender_logits, emotion_logits = predictions[0], predictions[1], predictions[2]  # Extract logits for each task  
        # #predictions = [age_logits, gender_logits, emotion_logits]
        
        # # Extract predictions for each task
        # age_pred = np.argmax(predictions[0])
        # gender_pred = np.argmax(predictions[1])
        # emotion_pred = np.argmax(predictions[2])
        
        # # Display predictions on the frame
        # cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        # cv2.putText(frame, f"Age: {age_labels[age_pred]}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # cv2.putText(frame, f"Gender: {gender_labels[gender_pred]}", (x, y+height+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # cv2.putText(frame, f"Emotion: {emotion_labels[emotion_pred]}", (x, y+height+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the frame
    cv2.imshow("Live Facial Analysis", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()