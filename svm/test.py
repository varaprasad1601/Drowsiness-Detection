import cv2
import pickle
import os

folder_path = "C:/Users/SHAIK BAJI BABA/3D Objects/test_data/"

for filename in os.listdir(folder_path):
    # Read the image file
    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64,64))

    # Flatten the image into a one-dimensional array
    image = image.reshape(1, -1)

    # Load the trained SVM model from the file
    with open('svm_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    # Use the trained SVM model to make a prediction on the image
    prediction = clf.predict(image)

    # Print the predicted label
    print(filename,"--->Prediction:", prediction[0])
