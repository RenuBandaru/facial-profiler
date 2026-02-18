# Merging two datasets (UTK-Face and RAF) into a single dataset for training and testing.
import os
import pandas as pd

# Reading the first dataset (UTK-Face)
utk_path = r"source_data/UTK-Face"

utk_rows = []

for root, dirs, files in os.walk(utk_path):
    for file in files:
        if file.endswith('.jpg'):
            try:
                parts = file.split('_')     # Split the filename to extract age, gender, and ethnicity

                age = int(parts[0])         # Age is the first part of the filename
                gender = int(parts[1])      # Gender is the second part of the filename
                race = int(parts[2])        # Ethnicity is the third part of the filename

                full_path = os.path.join(root, file)  # Get the full path of the image

                utk_rows.append({
                    "image_path": full_path, # storing the full path of the image for later use
                    "age": age,
                    "gender": gender,
                    "Race": race
                })

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                # skip the file if there's an error in processing
                continue

# Create a DataFrame from the list of rows
utk_df = pd.DataFrame(utk_rows)
utk_df.to_csv('utk_face_labels.csv', index=False)

print("UTK-Face dataset processed and saved to 'utk_face_labels.csv'.")
print(utk_df.head())

# Reading the second dataset (RAF)
raf_path = r"source_data/raf/DATASET/train"

raf_rows = []

for emotion_folder in os.listdir(raf_path):
    
    emotion_path = os.path.join(raf_path, emotion_folder) # Get the path of the emotion folder
    
    if os.path.isdir(emotion_path): # Check if it's a directory
        emotion_label = int(emotion_folder) # The emotion label is the name of the folder

        for file in os.listdir(emotion_path):
            if file.endswith(".jpg"):
                try:
                    full_path = os.path.join(emotion_path, file) # Get the full path of the image

                    raf_rows.append({
                        "image_path": full_path,     # Store the full path of the image
                        "emotion": emotion_label
                    })

                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    # skip the file if there's an error in processing
                    continue

# Create a DataFrame from the list of rows
raf_df = pd.DataFrame(raf_rows)
raf_df.to_csv('raf_labels.csv', index=False)

print("RAF dataset processed and saved to 'raf_labels.csv'.")
print(raf_df.head())