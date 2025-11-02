"""
Test script for evaluating the trained facial expression recognition model
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import argparse
from utils import (load_fer2013_data, evaluate_model, EMOTION_LABELS,
                  preprocess_image, download_haar_cascade)

def test_model_on_dataset(model_path, data_dir):
    """Test model on the test dataset"""
    print("Loading test data...")
    (_, _), (test_data, test_labels) = load_fer2013_data(data_dir)
    
    print("Loading model...")
    model = load_model(model_path)
    
    print("\\nEvaluating model on test dataset...")
    test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed evaluation
    predictions, true_labels = evaluate_model(model, test_data, test_labels)
    
    return model, test_accuracy

def test_model_on_image(model_path, image_path):
    """Test model on a single image"""
    # Load model
    model = load_model(model_path)
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    # Initialize face cascade
    cascade_path = download_haar_cascade()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        print("No faces detected in the image")
        return
    
    # Process each face
    for i, (x, y, w, h) in enumerate(faces):
        # Extract face
        face = gray[y:y+h, x:x+w]
        
        # Preprocess face
        processed_face = preprocess_image(face)
        
        # Predict emotion
        predictions = model.predict(processed_face, verbose=0)
        emotion_idx = np.argmax(predictions)
        confidence = predictions[0][emotion_idx]
        emotion_label = EMOTION_LABELS[emotion_idx]
        
        print(f"Face {i+1}: {emotion_label} (confidence: {confidence:.3f})")
        
        # Draw results on image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f"{emotion_label}: {confidence:.2f}", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display result
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Emotion Detection Results")
    plt.axis('off')
    plt.show()

def test_model_on_webcam(model_path, duration=30):
    """Test model on webcam for a specific duration"""
    from real_time_detection import RealTimeEmotionDetector
    
    print(f"Testing model on webcam for {duration} seconds...")
    detector = RealTimeEmotionDetector(model_path=model_path, use_pretrained=False)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    import time
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        processed_frame = detector.process_frame(frame)
        
        # Add timer
        elapsed = int(time.time() - start_time)
        remaining = duration - elapsed
        cv2.putText(processed_frame, f"Time remaining: {remaining}s", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Model Testing', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    fps = frame_count / duration
    print(f"Average FPS: {fps:.2f}")

def benchmark_model_performance(model_path):
    """Benchmark model inference time"""
    import time
    
    model = load_model(model_path)
    
    # Create dummy input
    dummy_input = np.random.random((1, 48, 48, 1)).astype(np.float32)
    
    # Warm up the model
    for _ in range(10):
        _ = model.predict(dummy_input, verbose=0)
    
    # Benchmark
    num_runs = 100
    start_time = time.time()
    
    for _ in range(num_runs):
        _ = model.predict(dummy_input, verbose=0)
    
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_inference_time
    
    print(f"\\nModel Performance Benchmark:")
    print(f"Average inference time: {avg_inference_time*1000:.2f} ms")
    print(f"Theoretical max FPS: {fps:.2f}")

def compare_with_baseline(model_path, data_dir):
    """Compare custom model with baseline methods"""
    print("Comparing with baseline methods...")
    
    # Load test data
    (_, _), (test_data, test_labels) = load_fer2013_data(data_dir)
    
    # Test custom model
    custom_model = load_model(model_path)
    custom_loss, custom_accuracy = custom_model.evaluate(test_data, test_labels, verbose=0)
    
    print(f"\\nComparison Results:")
    print(f"Custom CNN Model: {custom_accuracy:.4f}")
    
    # Compare with random baseline
    random_accuracy = 1.0 / 7  # Random guess for 7 classes
    print(f"Random Baseline: {random_accuracy:.4f}")
    
    # Performance improvement
    improvement = (custom_accuracy - random_accuracy) / random_accuracy * 100
    print(f"Improvement over random: {improvement:.2f}%")

def generate_test_report(model_path, data_dir, output_file="test_report.txt"):
    """Generate comprehensive test report"""
    print("Generating comprehensive test report...")
    
    with open(output_file, 'w') as f:
        f.write("FACIAL EXPRESSION RECOGNITION MODEL TEST REPORT\\n")
        f.write("=" * 50 + "\\n\\n")
        
        # Model information
        model = load_model(model_path)
        f.write(f"Model Path: {model_path}\\n")
        f.write(f"Model Parameters: {model.count_params():,}\\n")
        f.write(f"Input Shape: {model.input_shape}\\n")
        f.write(f"Output Classes: {len(EMOTION_LABELS)}\\n")
        f.write(f"Emotions: {list(EMOTION_LABELS.values())}\\n\\n")
        
        # Performance metrics
        print("Evaluating on test dataset...")
        (_, _), (test_data, test_labels) = load_fer2013_data(data_dir)
        test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=0)
        
        f.write("PERFORMANCE METRICS:\\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\\n")
        f.write(f"Test Loss: {test_loss:.4f}\\n\\n")
        
        # Benchmark performance
        print("Benchmarking inference speed...")
        import time
        dummy_input = np.random.random((1, 48, 48, 1)).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            _ = model.predict(dummy_input, verbose=0)
        
        # Benchmark
        num_runs = 100
        start_time = time.time()
        for _ in range(num_runs):
            _ = model.predict(dummy_input, verbose=0)
        avg_time = (time.time() - start_time) / num_runs
        
        f.write("PERFORMANCE BENCHMARK:\\n")
        f.write(f"Average Inference Time: {avg_time*1000:.2f} ms\\n")
        f.write(f"Theoretical Max FPS: {1.0/avg_time:.2f}\\n\\n")
        
        # Model architecture summary
        f.write("MODEL ARCHITECTURE:\\n")
        model.summary(print_fn=lambda x: f.write(x + '\\n'))
    
    print(f"Test report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Test Facial Expression Recognition Model')
    parser.add_argument('--model', default='models/emotion_model.h5', 
                       help='Path to trained model')
    parser.add_argument('--data_dir', default='data/fer2013', 
                       help='Path to test dataset')
    parser.add_argument('--test_type', choices=['dataset', 'image', 'webcam', 'benchmark', 'report'], 
                       default='dataset', help='Type of test to run')
    parser.add_argument('--image_path', help='Path to test image')
    parser.add_argument('--duration', type=int, default=30, 
                       help='Duration for webcam test (seconds)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first using train_model.py")
        return
    
    if args.test_type == 'dataset':
        if not os.path.exists(args.data_dir):
            print(f"Error: Dataset directory not found: {args.data_dir}")
            return
        test_model_on_dataset(args.model, args.data_dir)
        compare_with_baseline(args.model, args.data_dir)
        
    elif args.test_type == 'image':
        if not args.image_path:
            print("Error: Please provide --image_path for image testing")
            return
        test_model_on_image(args.model, args.image_path)
        
    elif args.test_type == 'webcam':
        test_model_on_webcam(args.model, args.duration)
        
    elif args.test_type == 'benchmark':
        benchmark_model_performance(args.model)
        
    elif args.test_type == 'report':
        if not os.path.exists(args.data_dir):
            print(f"Error: Dataset directory not found: {args.data_dir}")
            return
        generate_test_report(args.model, args.data_dir)

if __name__ == "__main__":
    main()