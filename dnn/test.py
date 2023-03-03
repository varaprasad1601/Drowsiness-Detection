import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Load the model
model = load_model('dnn_model.h5')

# Define the path to the test images folder
test_dir = "C:/Users/SHAIK BAJI BABA/3D Objects/test_data"

# Loop through each image file in the test directory
for file in os.listdir(test_dir):
    # Load the image using Keras' load_img function
    img = load_img(os.path.join(test_dir, file), target_size=(24, 24), color_mode='grayscale')
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    # Reshape the image to match the input shape of the model
    img_array = img_array.reshape(1, 24, 24, 1)
    # Make a prediction using the model
    prediction = model.predict(img_array)
    print("prediction :--->",prediction)
    # Get the predicted class index
    predicted_class = np.argmax(prediction)
    # Print the predicted class label
    if predicted_class == 0:
        print(file, ': Open Eyes')
    else:
        print(file, ': Closed Eyes')
