import numpy as np
import pandas as pd
import cv2
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns

def load_emnist_csv(file_path, n_samples=1000):
    data = pd.read_csv(file_path)
    data = data.sample(n=n_samples, random_state=42)

    labels = data.iloc[:, 0].values
    images = data.iloc[:, 1:].values
    images = images.reshape(-1, 28, 28).astype(np.uint8)
    return images, labels

def extract_hog_features(images, orientations=6, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    hog_features = []
    for image in images:
        image = cv2.resize(image, (28, 28))
        hog_feature = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block, visualize=False)
        hog_features.append(hog_feature)
    return np.array(hog_features)


def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.savefig('confusion_matrix.png')
    plt.close()

file_path = 'EMNIST Dataset/emnist-balanced-train.csv'
images, labels = load_emnist_csv(file_path, n_samples=1000)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

best_model = svm.SVC(kernel='linear', C=5, gamma='scale')

loo = LeaveOneOut()

y_true = []
y_pred = []

print("Starting LOOCV...")
for train_index, test_index in tqdm(loo.split(images), total=len(images), desc="LOOCV Progress"):
    X_train, X_test = images[train_index], images[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    X_train_hog = extract_hog_features(X_train, orientations=10, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    X_test_hog = extract_hog_features(X_test, orientations=10, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    
    best_model.fit(X_train_hog, y_train)
    prediction = best_model.predict(X_test_hog)
    
    y_true.append(y_test[0])
    y_pred.append(prediction[0])

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f'Accuracy (LOOCV): {accuracy * 100:.2f}%')
print(f'Precision (LOOCV): {precision:.2f}')
print(f'F1 Score (LOOCV): {f1:.2f}')

cm = confusion_matrix(y_true, y_pred)
classes = label_encoder.classes_
plot_confusion_matrix(cm, classes)

# Use emnist dataset