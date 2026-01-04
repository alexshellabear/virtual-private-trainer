import cv2
import numpy as np
import argparse
import os
import sys
from ultralytics import YOLO

# Ensure we can import from utils
sys.path.append(os.getcwd())
from utils.rep_counter import PoseClassifier

def train_from_video(video_path, exercise_name, state_labels):
    """
    video_path: Path to the MP4 file.
    exercise_name: Name of the exercise (e.g., 'squat').
    state_labels: List of state names (e.g., ['standing', 'squatting']).
    """
    print(f"--- Training {exercise_name} ---")
    print(f"Labels: {state_labels}")
    print("Controls:")
    for i, label in enumerate(state_labels):
        print(f"  Hold '{i+1}' to record '{label}'")
    print("  Press 'q' to finish and save.")
    print("  Press 'space' to pause/play.")

    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    # Initialize Models
    model = YOLO("yolo11n-pose.pt")
    classifier = PoseClassifier()
    
    cap = cv2.VideoCapture(video_path)
    training_data = []
    frame_count = 0
    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break
            frame_count += 1
        
        # Resize for easier viewing if 4k
        display_frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
        
        # Run Inference
        results = model(frame, verbose=False)
        
        # Draw instructions
        cv2.putText(display_frame, f"Frame: {frame_count} | Samples: {len(training_data)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        current_kpts = None
        
        # Extract Keypoints (Assume 1 person or take highest confidence)
        for r in results:
            if r.keypoints is not None and r.boxes is not None:
                # Find box with highest confidence
                confidences = r.boxes.conf.cpu().numpy()
                if len(confidences) > 0:
                    best_idx = np.argmax(confidences)
                    current_kpts = r.keypoints.xy[best_idx].cpu().numpy()
                    
                    # Draw skeleton on display frame
                    # Note: We need to scale keypoints to display_frame size for drawing
                    # But for training we use original keypoints
                    for kp in current_kpts:
                        x, y = int(kp[0] * 0.7), int(kp[1] * 0.7)
                        if x > 0 and y > 0:
                            cv2.circle(display_frame, (x, y), 3, (0, 0, 255), -1)

        cv2.imshow('Training Station', display_frame)
        
        key = cv2.waitKey(30 if not paused else 100) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        
        # Check for number keys (1-9)
        if current_kpts is not None:
            for i, label in enumerate(state_labels):
                # ord('1') is 49
                if key == ord(str(i + 1)):
                    print(f"Recorded: {label}")
                    training_data.append((current_kpts, label))
                    cv2.putText(display_frame, f"RECORDING: {label}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Training Station', display_frame)

    cap.release()
    cv2.destroyAllWindows()

    if len(training_data) > 0:
        print(f"Training complete. {len(training_data)} samples collected.")
        save = input("Save this model? (y/n): ").lower()
        if save == 'y':
            classifier.train_new_model(exercise_name, training_data)
            print(f"Model for '{exercise_name}' saved successfully!")
    else:
        print("No data collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pose Classifier from Video")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("name", help="Name of the exercise (e.g. 'squat')")
    parser.add_argument("--states", help="Comma separated states (default: 'standing,active')", default="standing,active")
    
    args = parser.parse_args()
    states = [s.strip() for s in args.states.split(',')]
    
    train_from_video(args.video, args.name, states)