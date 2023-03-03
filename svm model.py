# Import necessary libraries
from sklearn import datasets, svm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import os

# Load the image dataset
data_path = "C:/Users/SHAIK BAJI BABA/3D Objects/dataset1/train"
categories = os.listdir(data_path)
images = []
labels = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64,64)) # Resize image to 64x64 pixels
        images.append(image)
        labels.append(category)

# Convert images and labels into numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)
print("xtrain\n",X_train,"len :",len(X_train))
print("ytrain\n",y_train,"len :",len(y_train))
print("xtest\n",X_test,"len :",len(X_test))
print("ytest\n",y_test,"len :",len(y_test))

# Flatten images into one-dimensional arrays
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Initialize the SVM model
clf = svm.SVC(kernel='linear', C=1, gamma='scale')

# Train the SVM model on the training set
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

"""predictions_df = pd.DataFrame(, columns=['predictions'])
predictions_df.to_csv('predictions.csv', index=False)"""


# Save the trained SVM model to a file
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Evaluate the accuracy of the SVM model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


