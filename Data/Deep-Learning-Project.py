#!/usr/bin/env python
# coding: utf-8

# # 1) Purpose of This Step (High-level)
# 
# Goal:
# Create one unified dataset representation that combines:
# 
# FER-2013 → Emotion labels
# 
# UTKFace → Age + Gender labels
# 
# ❗ Important:
# 
# We are NOT combining labels per image
# 
# We are NOT training anything
# 
# We are creating a single CSV manifest that supports multi-task learning later

# # 2️) Imports (Notebook Cell)

# In[20]:


# Standard libraries for file handling
import os
import glob

# Data handling
import pandas as pd


# # 3) Define Dataset Paths

# In[35]:


# Root folder for FER-2013
# Expected structure:
# fer2013/
#   train/angry, happy, sad, ...
#   test/angry, happy, sad, ...
FER_ROOT = r"C:\Users\bensa\Desktop\ETUDES\Deep Learning\Project\Fer2013"

# Root folder for UTKFace
# Contains images named:
# age_gender_race_timestamp.jpg
UTK_ROOT = r"C:\Users\bensa\Desktop\ETUDES\Deep Learning\Project\UTK-Face\Part1"

# Verify folders exist
print("FER train exists:", os.path.isdir(os.path.join(FER_ROOT, "train")))
print("UTKFace exists:", os.path.isdir(UTK_ROOT))


# # 4) Define Emotion Labels (FER-2013)
# 
# FER-2013 stores labels as folder names, not numbers.
# We convert them to numeric class IDs.

# In[36]:


# Emotion categories used in FER-2013
FER_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Map each emotion to a numeric label
fer_to_idx = {emotion: idx for idx, emotion in enumerate(FER_CLASSES)}


# # 5) Build FER-2013 Dataset Entries
# 
#  Each FER image:
# 
# Has emotion
# 
# Does not have age or gender

# In[37]:


def build_fer_rows(fer_root, split="train"):
    """
    Create dataset rows for FER-2013 images.

    Each row includes:
    - image path
    - emotion label
    - placeholders for age and gender
    - mask values indicating which labels exist
    """
    rows = []
    split_dir = os.path.join(fer_root, split)

    for emotion in FER_CLASSES:
        emotion_dir = os.path.join(split_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue

        for img_path in glob.glob(os.path.join(emotion_dir, "*.*")):
            rows.append({
                "path": img_path,
                "emotion": fer_to_idx[emotion],
                "age": -1,              # not available
                "gender": -1,           # not available
                "has_emotion": 1,
                "has_age": 0,
                "has_gender": 0,
                "source": f"FER_{split}"
            })
    return rows


# # 6) Parse UTKFace Filenames
# 
# UTKFace encodes labels inside the filename.
# 
# Example:
# 25_0_0_20170116174525125.jpg
# → age = 25, gender = 0 (male)

# In[38]:


import os
import glob



def parse_utk_filename(path):
    """
    Parse age and gender from UTKFace filenames.

    Expected filename format:
    age_gender_race_timestamp.jpg

    Example:
    25_0_0_20170116174525125.jpg
    """
    filename = os.path.basename(path)
    print("Filename:", filename)

    # Remove all extensions (handles .jpg, .jpg.chip.jpg, etc.)
    stem = filename.split(".")[0]
    print("Filename without extension:", stem)

    parts = stem.split("_")
    print("Split parts:", parts)

    if len(parts) < 2:
        print("❌ Not enough parts — skipping\n")
        return None

    try:
        age = int(parts[0])
        gender = int(parts[1])
        print(f"✅ Parsed → age: {age}, gender: {gender}\n")
        return age, gender
    except ValueError:
        print("❌ Failed to convert age/gender to int — skipping\n")
        return None


# --- DISPLAY OUTPUTS ---
print("🔍 Scanning UTKFace directory...\n")

files = glob.glob(os.path.join(UTK_ROOT, "*.*"))
print(f"Total files found: {len(files)}\n")

# Show parsing results for ALL files (warning: many prints if dataset is large)
for f in files:
    parse_utk_filename(f)


# # 7) Build UTKFace Dataset Entries
# 
# Each UTKFace image:
# 
# Has age + gender
# 
# Does not have emotion

# In[39]:


def build_utk_rows(utk_root):
    """
    Create dataset rows for UTKFace images.

    These rows include age and gender,
    but no emotion labels.
    """
    rows = []

    for img_path in glob.glob(os.path.join(utk_root, "*.*")):
        parsed = parse_utk_filename(img_path)
        if parsed is None:
            continue

        age, gender = parsed
        rows.append({
            "path": img_path,
            "emotion": -1,          # not available
            "age": age,
            "gender": gender,
            "has_emotion": 0,
            "has_age": 1,
            "has_gender": 1,
            "source": "UTKFace"
        })

    return rows


# # 8) Create the Merged Dataset Manifest (CSV)

# In[40]:


# Collect rows from both datasets
rows = []
rows.extend(build_fer_rows(FER_ROOT, "train"))
rows.extend(build_fer_rows(FER_ROOT, "test"))
rows.extend(build_utk_rows(UTK_ROOT))

# Create DataFrame
merged_df = pd.DataFrame(rows)

# Save merged dataset manifest
merged_df.to_csv("merged_face_dataset.csv", index=False)

# Inspect result
merged_df.head(), merged_df["source"].value_counts()


# The FER-2013 and UTKFace datasets were merged by creating a unified dataset manifest rather than combining labels at the image level. Each image is represented by its file path and associated labels, with missing labels explicitly marked. FER-2013 samples provide emotion annotations, while UTKFace samples provide age and gender annotations. Binary mask fields indicate which labels are available for each image, enabling flexible multi-task learning in later stages without introducing label noise or incorrect supervision.

# In[ ]:




