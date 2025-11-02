"""
Utility functions for facial expression recognition
"""
import numpy as np
import cv2
import os
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Emotion labels
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust', 
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

def create_directory_structure():
    """Create necessary directories for the project"""
    directories = [
        'data/fer2013/train',
        'data/fer2013/test', 
        'models',
        'haarcascades',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directory structure created successfully!")

def download_haar_cascade():
    """Download Haar cascade file if not present"""
    cascade_path = 'haarcascades/haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        import urllib.request
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
        urllib.request.urlretrieve(url, cascade_path)
        print("Haar cascade downloaded successfully!")
    return cascade_path

def preprocess_image(image, target_size=(48, 48)):
    """Preprocess image for model input"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    
    return image

def load_fer2013_data(data_dir):
    """Load FER2013 dataset from directory structure"""
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    # Load training data
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        for emotion_idx, emotion_name in EMOTION_LABELS.items():
            emotion_dir = os.path.join(train_dir, str(emotion_idx))
            if os.path.exists(emotion_dir):
                for filename in os.listdir(emotion_dir):
                    if filename.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(emotion_dir, filename)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (48, 48))
                            train_data.append(img)
                            train_labels.append(emotion_idx)
    
    # Load test data
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        for emotion_idx, emotion_name in EMOTION_LABELS.items():
            emotion_dir = os.path.join(test_dir, str(emotion_idx))
            if os.path.exists(emotion_dir):
                for filename in os.listdir(emotion_dir):
                    if filename.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(emotion_dir, filename)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (48, 48))
                            test_data.append(img)
                            test_labels.append(emotion_idx)
    
    # Convert to numpy arrays
    train_data = np.array(train_data, dtype='float32') / 255.0
    train_labels = to_categorical(train_labels, num_classes=7)
    test_data = np.array(test_data, dtype='float32') / 255.0  
    test_labels = to_categorical(test_labels, num_classes=7)
    
    # Reshape for CNN input
    train_data = train_data.reshape(-1, 48, 48, 1)
    test_data = test_data.reshape(-1, 48, 48, 1)
    
    return (train_data, train_labels), (test_data, test_labels)

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.show()

def evaluate_model(model, test_data, test_labels):
    """Evaluate model and show metrics"""
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(true_classes, predicted_classes, 
                              target_names=list(EMOTION_LABELS.values())))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(EMOTION_LABELS.values()),
                yticklabels=list(EMOTION_LABELS.values()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('results/confusion_matrix.png')
    plt.show()
    
    return predicted_classes, true_classes

def detect_faces(image, cascade_classifier):
    """Detect faces in image using Haar cascade"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def predict_emotion(model, face_image):
    """Predict emotion from face image"""
    processed_face = preprocess_image(face_image)
    predictions = model.predict(processed_face, verbose=0)
    emotion_idx = np.argmax(predictions)
    confidence = predictions[0][emotion_idx]
    emotion_label = EMOTION_LABELS[emotion_idx]
    
    return emotion_label, confidence

def draw_emotion_label(image, x, y, w, h, emotion, confidence):
    """Draw emotion label on image"""
    # Draw rectangle around face
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Draw emotion label
    label = f"{emotion}: {confidence:.2f}"
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, (255, 0, 0), 2)
    
    return image

def get_model_summary_info():
    """Get information about the model architecture"""
    info = {
        'input_shape': (48, 48, 1),
        'num_classes': 7,
        'emotions': list(EMOTION_LABELS.values()),
        'architecture': 'CNN with 4 conv layers + 2 dense layers'
    }
    return info

if __name__ == "__main__":
    # Create directory structure when run directly
    create_directory_structure()
    download_haar_cascade()
    print("Setup completed successfully!")