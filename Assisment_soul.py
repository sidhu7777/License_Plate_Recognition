#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pytesseract

# Set Tesseract executable path manually
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# In[3]:


get_ipython().system('pip install pytesseract')


# In[13]:


import os
import pandas as pd
from PIL import Image
import zipfile
import matplotlib.pyplot as plt

# Function to extract images from zip files
def extract_images(zip_path, extract_to):
    """Extract images from a zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Images extracted to {extract_to}")
    except Exception as e:
        print(f"Error extracting images: {e}")

# Function to process CSV for bounding boxes (from detection phase)
def process_csv(csv_path):
    """Process the CSV file to get bounding box coordinates."""
    df = pd.read_csv(csv_path)
    print(f"Total entries in CSV: {len(df)}")
    return df

# Function to visualize the images with bounding boxes
def visualize_image_with_bbox(image_path, xmin, ymin, xmax, ymax):
    """Visualize an image with a bounding box."""
    img = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                      fill=False, edgecolor='red', linewidth=2))
    plt.title('Image with License Plate Bounding Box')
    plt.axis('off')
    plt.show()

# Function to crop and detect license plates
def detect_license_plate(image_path, xmin, ymin, xmax, ymax):
    """Detect and crop the license plate region from an image."""
    img = Image.open(image_path)
    plate_img = img.crop((xmin, ymin, xmax, ymax))
    return plate_img

# Function to process the first 10 detection images
def process_first_10_detection_images(detection_zip, detection_csv, extract_dir):
    # Extract images from the detection zip file
    extract_images(detection_zip, extract_dir)

    # Process the CSV for the annotations
    df_detection = process_csv(detection_csv)

    # Loop through the first 10 images from the detection CSV
    for index in range(10):  # Limiting to first 10 images
        if index >= len(df_detection):
            break

        # Get the image and bounding box from the detection CSV
        detection_row = df_detection.iloc[index]
        img_file_detection = detection_row['img_id']
        xmin, ymin, xmax, ymax = detection_row['xmin'], detection_row['ymin'], detection_row['xmax'], detection_row['ymax']

        # Full path to the detection image
        img_path_detection = os.path.join(extract_dir, 'license_plates_detection_train', img_file_detection)

        # Perform detection: visualize image with bounding box
        print(f"Detection - Image: {img_file_detection}")
        visualize_image_with_bbox(img_path_detection, xmin, ymin, xmax, ymax)

        # Crop and display the detected license plate
        cropped_plate = detect_license_plate(img_path_detection, xmin, ymin, xmax, ymax)
        cropped_plate.show()  # Display the cropped license plate

# Main function to run the detection pipeline for the first 10 images
if __name__ == "__main__":
    # Paths for detection data
    detection_zip = r"C:\Users\91832\Desktop\Assignment_soul\Licplatesdetection_train.zip"
    detection_csv = r"C:\Users\91832\Desktop\Assignment_soul\Licplatesdetection_train.csv"
    extract_dir = r"C:\Users\91832\Desktop\Assignment_soul\extracted_images"
    
    # Run the detection pipeline for the first 10 images
    process_first_10_detection_images(detection_zip, detection_csv, extract_dir)


# In[5]:


import os
import pandas as pd
from PIL import Image
import pytesseract
import zipfile

# Function to extract images from zip files
def extract_images(zip_path, extract_to):
    """Extract images from a zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Images extracted to {extract_to}")
    except Exception as e:
        print(f"Error extracting images: {e}")

# Function to process CSV for plate text (from recognition phase)
def process_csv(csv_path):
    """Process the CSV file to get the text annotation for license plates."""
    df = pd.read_csv(csv_path)
    print(f"Total entries in CSV: {len(df)}")
    return df

# Function to preprocess the image (grayscale, thresholding, denoising)
def preprocess_image(image_path):
    """Preprocess the image before passing it to Tesseract for OCR."""
    img = Image.open(image_path)
    gray = img.convert('L')  # Convert image to grayscale
    return gray

# Function to recognize text using Tesseract OCR
def recognize_license_plate(plate_img):
    """Perform OCR on the license plate to recognize text."""
    try:
        plate_text = pytesseract.image_to_string(plate_img, config='--psm 8 --oem 3')
        return plate_text.strip()  # Clean the recognized text
    except Exception as e:
        print(f"Error in recognition: {e}")
        return ""

# Function to process the first 10 recognition images
def process_first_10_recognition_images(recognition_zip, recognition_csv, extract_dir):
    # Extract images from the recognition zip file
    extract_images(recognition_zip, extract_dir)

    # Process the CSV for the annotations
    df_recognition = process_csv(recognition_csv)

    # Loop through the first 10 images from the recognition CSV
    for index in range(10):  # Limiting to first 10 images
        if index >= len(df_recognition):
            break

        # Get the image and actual plate text from the recognition CSV
        recognition_row = df_recognition.iloc[index]
        img_file_recognition = recognition_row['img_id']
        actual_plate_text = recognition_row['text']

        # Full path to the recognition image
        img_path_recognition = os.path.join(extract_dir, 'license_plates_recognition_train', img_file_recognition)

        # Perform OCR on the license plate from recognition.zip
        print(f"Recognition - Image: {img_file_recognition}")
        preprocessed_img_recognition = preprocess_image(img_path_recognition)
        recognized_plate_text = recognize_license_plate(preprocessed_img_recognition)
        print(f"Detected Plate Text: {recognized_plate_text}, Actual Plate Text: {actual_plate_text}")

# Main function to run the recognition pipeline for the first 10 images
if __name__ == "__main__":
    # Paths for recognition data
    recognition_zip = r"C:\Users\91832\Desktop\Assignment_soul\Licplatesrecognition_train.zip"
    recognition_csv = r"C:\Users\91832\Desktop\Assignment_soul\Licplatesrecognition_train.csv"
    extract_dir = r"C:\Users\91832\Desktop\Assignment_soul\extracted_images"
    
    # Run the recognition pipeline for the first 10 images
    process_first_10_recognition_images(recognition_zip, recognition_csv, extract_dir)


# In[6]:


import os
import pandas as pd
from PIL import Image
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to extract images from zip files
def extract_images(zip_path, extract_to):
    """Extract images from a zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Images extracted to {extract_to}")
    except Exception as e:
        print(f"Error extracting images: {e}")

# Function to process CSV for bounding boxes (from detection phase)
def process_csv(csv_path):
    """Process the CSV file to get bounding box coordinates."""
    df = pd.read_csv(csv_path)
    print(f"Total entries in CSV: {len(df)}")
    return df

# Function to load and preprocess images for training
def load_and_preprocess_image(image_path):
    """Load and preprocess the image for the model."""
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to a fixed size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to create the detection model
def create_detection_model():
    """Create a simple CNN model for license plate detection."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4)  # Output layer for bounding box coordinates (xmin, ymin, xmax, ymax)
    ])
    model.compile(optimizer='adam', loss='mse')  # Using MSE for regression (bounding box)
    return model

# Function to prepare data for model training
def prepare_data_for_training(detection_zip, detection_csv, extract_dir):
    """Prepare the data for training the detection model."""
    # Extract images from the detection zip file
    extract_images(detection_zip, extract_dir)

    # Process the CSV for the annotations
    df_detection = process_csv(detection_csv)

    images = []
    bboxes = []

    # Loop through the images and corresponding bounding boxes
    for _, row in df_detection.iterrows():
        img_file = row['img_id']
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

        # Full path to the detection image
        img_path = os.path.join(extract_dir, 'license_plates_detection_train', img_file)
        img_array = load_and_preprocess_image(img_path)
        
        images.append(img_array)
        bboxes.append([xmin, ymin, xmax, ymax])

    images = tf.convert_to_tensor(images)
    bboxes = tf.convert_to_tensor(bboxes)

    return images, bboxes

# Main function to train the detection model
def train_detection_model(detection_zip, detection_csv, extract_dir, model_save_path):
    """Train the license plate detection model."""
    # Prepare the data for training
    images, bboxes = prepare_data_for_training(detection_zip, detection_csv, extract_dir)

    # Create the model
    model = create_detection_model()

    # Train the model
    model.fit(images, bboxes, epochs=10, batch_size=32)

    # Save the trained model
    model.save(model_save_path)
    print(f"Model trained and saved successfully at {model_save_path}")

# Main entry point
if __name__ == "__main__":
    # Paths for detection data
    detection_zip = r"C:\Users\91832\Desktop\Assignment_soul\Licplatesdetection_train.zip"
    detection_csv = r"C:\Users\91832\Desktop\Assignment_soul\Licplatesdetection_train.csv"
    extract_dir = r"C:\Users\91832\Desktop\Assignment_soul\extracted_images"
    model_save_path = r"C:\Users\91832\Desktop\Assignment_soul\trained_detection_model.h5"
    
    # Train the detection model
    train_detection_model(detection_zip, detection_csv, extract_dir, model_save_path)


# In[7]:


get_ipython().system('pip install protobuf==3.20.3')


# In[ ]:





# In[8]:


import os
import pandas as pd
import numpy as np
import zipfile
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Function to extract images from zip files
def extract_images(zip_path, extract_to):
    """Extract images from a zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Images extracted to {extract_to}")
    except Exception as e:
        print(f"Error extracting images: {e}")

# Function to process CSV for plate text (from recognition phase)
def process_csv(csv_path):
    """Process the CSV file to get the text annotation for license plates."""
    df = pd.read_csv(csv_path)
    print(f"Total entries in CSV: {len(df)}")
    return df

# Function to preprocess the image (resize, normalization)
def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess the image (resize, grayscale, normalization)."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Function to prepare the data (extract images, process CSV)
def prepare_data_for_recognition(zip_file, csv_file, extract_dir):
    """Prepare data for recognition by extracting images and labels."""
    # Extract images
    extract_images(zip_file, extract_dir)
    
    # Process CSV file
    df = process_csv(csv_file)
    
    images = []
    labels = []
    
    for _, row in df.iterrows():
        img_file = row['img_id']
        label = row['text']
        
        # Full path to the image
        img_path = os.path.join(extract_dir, 'license_plates_recognition_train', img_file)
        
        if os.path.exists(img_path):
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(label)
        else:
            print(f"Image not found: {img_path}")
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Function to create the CNN model
def create_recognition_model(input_shape=(224, 224, 3), num_classes=36):
    """Create a CNN model for license plate recognition."""
    model = Sequential()
    
    # Convolutional Layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Flatten the results to feed into a fully connected layer
    model.add(Flatten())
    
    # Fully Connected Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    model.add(Dense(num_classes, activation='softmax'))  # Output layer with softmax for classification
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Main function to train the recognition model
def train_recognition_model(zip_file, csv_file, extract_dir, epochs=10, batch_size=32):
    """Train the license plate recognition model."""
    
    # Prepare data
    images, labels = prepare_data_for_recognition(zip_file, csv_file, extract_dir)
    
    # Convert labels to categorical (one-hot encoding)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_one_hot = to_categorical(labels_encoded)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)
    
    # Create model
    model = create_recognition_model(input_shape=(224, 224, 3), num_classes=len(label_encoder.classes_))
    
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    
    # Save the trained model
    model.save(r"C:\Users\91832\Desktop\Assignment_soul\trained_recognition_model.h5")
    print("Model saved successfully at 'trained_recognition_model.h5'")

# Run the model training
if __name__ == "__main__":
    # Paths for recognition data
    recognition_zip = r"C:\Users\91832\Desktop\Assignment_soul\Licplatesrecognition_train.zip"
    recognition_csv = r"C:\Users\91832\Desktop\Assignment_soul\Licplatesrecognition_train.csv"
    extract_dir = r"C:\Users\91832\Desktop\Assignment_soul\extracted_images"
    
    # Train the recognition model
    train_recognition_model(recognition_zip, recognition_csv, extract_dir)


# In[4]:


"""this is the code to see the file format in the zip file """
"""""import zipfile

# Path to the ZIP file
zip_path = r"C:\Users\91832\Desktop\Assignment_soul\test (1).zip"

# Open the ZIP file and list all file paths inside it
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_files = zip_ref.namelist()
    for file in zip_files:
        print(file)


# In[12]:


test_zip = r"C:\Users\91832\Desktop\Assignment_soul\test (1).zip"
detection_model_path = r"C:\Users\91832\Desktop\Assignment_soul\trained_detection_model.h5"
recognition_model_path = r"C:\Users\91832\Desktop\Assignment_soul\trained_recognition_model.h5"


# In[ ]:


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import zipfile

# Function to extract images from a ZIP file
def extract_images(zip_path, extract_dir):
    """Extracts all images from the given zip file to the specified directory."""
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Images extracted to {extract_dir}")

# Function to preprocess an image for model input
def preprocess_image(image, target_size=(224, 224)):
    """Resize and normalize an image for model input."""
    resized = cv2.resize(image, target_size)
    normalized = resized / 255.0
    return normalized

# Function to crop a license plate based on bounding box
def crop_license_plate(image, box):
    """Crop the region of the license plate from the image."""
    x1, y1, x2, y2 = map(int, box)
    h, w, _ = image.shape
    
    # Ensure the box is within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

# Main testing pipeline
def test_pipeline(zip_path, detection_model_path, recognition_model_path, extract_dir):
    """Pipeline to test detection and recognition on images."""
    # Step 1: Extract images
    extract_images(zip_path, extract_dir)

    # Step 2: Load models
    detection_model = load_model(detection_model_path)
    recognition_model = load_model(recognition_model_path)
    print("Models loaded successfully.")

    # Character mapping for recognition output
    char_map = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Adjust as per the trained model's character set

    # Step 3: Process images
    image_files = [os.path.join(root, file) 
                   for root, _, files in os.walk(extract_dir) 
                   for file in files if file.lower().endswith(('.jpg', '.png'))]
    
    for img_path in image_files:
        print(f"Processing image: {img_path}")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Unable to read image: {img_path}")
            continue

        # Preprocess image for detection model
        input_image = preprocess_image(image)
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

        # Step 4: Perform detection
        detections = detection_model.predict(input_image)

        # Log raw detection outputs
        print(f"Raw detection output for {img_path}: {detections[0]}")

        # Assuming the model outputs normalized bounding boxes [x1, y1, x2, y2] in range [0, 1]
        x1, y1, x2, y2 = detections[0]

        # Sanity check: Ensure valid predictions
        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1 or x1 >= x2 or y1 >= y2:
            print(f"Invalid raw bounding box from model: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            continue

        # Scale bounding box back to the original image size
        height, width, _ = image.shape
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)

        # Clip bounding box to image dimensions
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        # Log scaled and clipped bounding box
        print(f"Scaled and clipped bounding box for {img_path}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        # Validate bounding box dimensions
        if (x2 - x1) < 10 or (y2 - y1) < 10:  # Skip overly small bounding boxes
            print(f"Bounding box too small after scaling: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            continue


        # Step 5: Crop license plate
        try:
            cropped_plate = crop_license_plate(image, [x1, y1, x2, y2])
            if cropped_plate is None or cropped_plate.size == 0:
                print("Cropped plate is invalid or empty. Skipping recognition.")
                continue

            # Save cropped plates for debugging
            debug_dir = os.path.join(extract_dir, "debug_crops")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, os.path.basename(img_path))
            cv2.imwrite(debug_path, cropped_plate)
            print(f"Cropped license plate saved to {debug_path}")
        except Exception as e:
            print(f"Error during cropping: {e}")
            continue

        # Step 6: Preprocess cropped license plate for recognition
        plate_image = preprocess_image(cropped_plate)
        plate_image = np.expand_dims(plate_image, axis=0)  # Add batch dimension

        # Step 7: Perform recognition
        recognition_result = recognition_model.predict(plate_image)

        # Log raw predictions
        print(f"Raw recognition output: {recognition_result}")

        try:
            # Decode recognized text using character map
            recognized_text = ''.join([char_map[idx] for idx in np.argmax(recognition_result, axis=-1)])
            print(f"Recognized text: {recognized_text}")
        except Exception as e:
            print(f"Error decoding recognition result: {e}")

        # Save image with bounding box for debugging
        try:
            debug_image = image.copy()
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            debug_boxes_dir = os.path.join(extract_dir, "debug_boxes")
            os.makedirs(debug_boxes_dir, exist_ok=True)
            debug_image_path = os.path.join(debug_boxes_dir, os.path.basename(img_path))
            cv2.imwrite(debug_image_path, debug_image)
            print(f"Image with bounding box saved to {debug_image_path}")
        except Exception as e:
            print(f"Error saving debug image with bounding box: {e}")

# File paths for testing
test_zip = r"C:\Users\91832\Desktop\Assignment_soul\test (1).zip"
detection_model_path = r"C:\Users\91832\Desktop\Assignment_soul\trained_detection_model.h5"
recognition_model_path = r"C:\Users\91832\Desktop\Assignment_soul\trained_recognition_model.h5"
extract_dir = r"C:\Users\91832\Desktop\Assignment_soul\extracted_test_images"

# Run the testing pipeline
test_pipeline(test_zip, detection_model_path, recognition_model_path, extract_dir)


# In[ ]:




