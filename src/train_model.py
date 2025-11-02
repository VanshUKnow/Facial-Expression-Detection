"""
Training script for facial expression recognition model
"""
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import argparse
import json
from utils import create_directory_structure, load_fer2013_data, plot_training_history, EMOTION_LABELS

class EmotionModel:
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build CNN model architecture"""
        self.model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block  
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            
            # Fully Connected Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def get_callbacks(self, model_path='models/best_emotion_model.h5'):
        """Define training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, train_data, train_labels, validation_data, validation_labels, 
              epochs=50, batch_size=64, use_augmentation=True):
        """Train the model"""
        
        # Data augmentation
        if use_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest'
            )
            
            # Fit the data generator
            datagen.fit(train_data)
            
            # Train with data augmentation
            history = self.model.fit(
                datagen.flow(train_data, train_labels, batch_size=batch_size),
                steps_per_epoch=len(train_data) // batch_size,
                epochs=epochs,
                validation_data=(validation_data, validation_labels),
                callbacks=self.get_callbacks(),
                verbose=1
            )
        else:
            # Train without data augmentation
            history = self.model.fit(
                train_data, train_labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(validation_data, validation_labels),
                callbacks=self.get_callbacks(),
                verbose=1
            )
        
        return history
    
    def save_model(self, model_path='models/emotion_model.h5', 
                   architecture_path='models/model_architecture.json'):
        """Save the trained model"""
        # Save the complete model
        self.model.save(model_path)
        
        # Save model architecture
        model_json = self.model.to_json()
        with open(architecture_path, 'w') as json_file:
            json_file.write(model_json)
            
        print(f"Model saved to {model_path}")
        print(f"Architecture saved to {architecture_path}")

def load_data_from_csv(csv_path):
    """Alternative method to load FER2013 data from CSV file"""
    import pandas as pd
    
    # Load the CSV file
    data = pd.read_csv(csv_path)
    
    # Extract pixels and emotions
    pixels = data['pixels'].tolist()
    emotions = data['emotion'].tolist()
    usage = data['Usage'].tolist()
    
    # Process images
    images = []
    for pixel_string in pixels:
        img = np.fromstring(pixel_string, sep=' ')
        img = img.reshape(48, 48, 1)
        img = img.astype('float32') / 255.0
        images.append(img)
    
    images = np.array(images)
    emotions = tf.keras.utils.to_categorical(emotions, num_classes=7)
    
    # Split data
    train_images = images[np.array(usage) == 'Training']
    train_emotions = emotions[np.array(usage) == 'Training']
    
    test_images = images[np.array(usage) == 'PublicTest']
    test_emotions = emotions[np.array(usage) == 'PublicTest']
    
    return (train_images, train_emotions), (test_images, test_emotions)

def main():
    parser = argparse.ArgumentParser(description='Train Facial Expression Recognition Model')
    parser.add_argument('--data_dir', default='data/fer2013', help='Path to dataset directory')
    parser.add_argument('--csv_path', default=None, help='Path to FER2013 CSV file (alternative to data_dir)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--augmentation', action='store_true', default=True, help='Use data augmentation')
    
    args = parser.parse_args()
    
    # Create directory structure
    create_directory_structure()
    
    print("Loading dataset...")
    if args.csv_path and os.path.exists(args.csv_path):
        # Load from CSV file
        (train_data, train_labels), (test_data, test_labels) = load_data_from_csv(args.csv_path)
        print(f"Loaded data from CSV: {args.csv_path}")
    else:
        # Load from directory structure
        (train_data, train_labels), (test_data, test_labels) = load_fer2013_data(args.data_dir)
        print(f"Loaded data from directory: {args.data_dir}")
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    # Create and build model
    emotion_model = EmotionModel()
    model = emotion_model.build_model()
    emotion_model.compile_model(learning_rate=args.learning_rate)
    
    # Print model summary
    print("\\nModel Architecture:")
    model.summary()
    
    # Train the model
    print("\\nStarting training...")
    history = emotion_model.train(
        train_data, train_labels,
        test_data, test_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_augmentation=args.augmentation
    )
    
    # Save the model
    emotion_model.save_model()
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test data
    print("\\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save training results
    results = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'emotions': EMOTION_LABELS
    }
    
    with open('results/training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\\nTraining completed successfully!")
    print("Model saved to 'models/emotion_model.h5'")
    print("Results saved to 'results/training_results.json'")

if __name__ == "__main__":
    main()