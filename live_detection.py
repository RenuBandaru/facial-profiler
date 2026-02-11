
#Opening the webcam and detecting faces in real-time using OpenCV's Haar Cascade classifier.
import cv2

# Load the Haar Cascade classifier for face detection
face_casecade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to detect features
def detect_features(frame): 
    #going to make the frame grayscale to make the detection easier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces =  face_casecade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) #detect faces (all of them), lower the number of neighbors, the lesser the quality of detection but more detections
            # you can also specify minSize and maxSize parameters to limit the size of detected faces
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) #draw rectangle around the face
        roi_gray = gray[y:y + h, x:x + w] #region of interest in grayscale
        roi_color = frame[y:y + h, x:x + w] #region of interest in color

        # Detecting smiles within the face region
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2) #draw rectangle around the smile

        # Detecting eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2) #draw rectangle around the eyes
        
    return frame

# Start video capture from the default webcam
stream = cv2.VideoCapture(0) # 0 is the default webcam, you can change it to 1 or 2 if you have multiple cameras or filepath to a video file

# checking if the stream is open
if not stream.isOpened():
    print("Error: Could not open video stream.")
    exit()

#Displaying the stream in a window
# we are going to display each frame in a window and we will also check for the 'q' key to exit the loop
while True:
    # Read a frame from the video capture
    ret, frame = stream.read()

    # If the frame was not read successfully, break the loop
    if not ret: #meaning the return could be false
        print("Error: Could not read frame or No more stream to read.")
        break

    # Detect features in the frame
    frame = detect_features(frame)
    
    # Display the frame
    cv2.imshow('Webcam Live Stream', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) == ord('q'): #0 is true if any key is pressed, 1 will let us choose a specifc key and the ord will cast the q to the specific ascii integer value of the key
        print("Exiting the live stream.")
        break

# Release the video capture and close all OpenCV windows
stream.release()
cv2.destroyAllWindows()