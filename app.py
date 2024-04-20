import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras_preprocessing.image import img_to_array
import cv2
import time
import tempfile
import os

# Load the emotion detection model
model = load_model("emotiondetector.h5")
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
gender_model = load_model("gender_model.h5")
age_model = load_model("age_model.h5")

# Labels for gender and age prediction
gender_ranges = ['male', 'female']
age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']

def predict_gender_age(image):


    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gender_img = cv2.resize(gray, (100, 100), interpolation = cv2.INTER_AREA)
    gender_image_array = np.array(gender_img)
    gender_input = np.expand_dims(gender_image_array, axis=0)
    output_gender=gender_ranges[np.argmax(gender_model.predict(gender_input))]

    age_image=cv2.resize(gray, (200, 200), interpolation = cv2.INTER_AREA)
    age_input = age_image.reshape(-1, 200, 200, 1)
    output_age = age_ranges[np.argmax(age_model.predict(age_input))]
    

    return output_gender, output_age

def predict_emotion(image):
    
    # Convert the image to grayscale, resize, and preprocess
    image = image.convert('L').resize((48, 48))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    pred = model.predict(img_array)
    pred_label = labels[np.argmax(pred)]
    return pred_label

def main():
    



    st.title("Emotion Detector App")
    st.write("Choose an option to detect emotion:")

    option = st.radio("", ("Upload Image", "Capture Live Image"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
            file_path = path
            img=cv2.imread(path)

            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            pred_emotion = predict_emotion(image)
            
            gender, age = predict_gender_age(img)

            st.write(f"Predicted Gender: {gender}")
            st.write(f"Predicted Age: {age}")
            st.write(f"Predicted Emotion: {pred_emotion}")

    elif option == "Capture Live Image":
        st.write("Capturing live image...")
        capture = cv2.VideoCapture(1)
        time.sleep(3)
        ret, frame = capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            st.image(image, caption='Live Image', use_column_width=True)
            pred_emotion = predict_emotion(image)
            st.write(f"Predicted Emotion: {pred_emotion}")
        else:
            st.write("Failed to capture image.")

    
if __name__ == "__main__":
    main()
