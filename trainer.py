import os
from ultralytics import YOLO
import json
import logging
import datetime
import time
import argparse
import threading
import requests
from dotenv import load_dotenv
import cv2
import pygame
from concurrent.futures import ThreadPoolExecutor

from utils.audio_coach import AudioCoach
from utils.rep_counter import RepCounter
from utils.clean_history import clean_workout_history
from utils.pose_validator import PoseValidator

# Load Environment Variables
load_dotenv()

# Configuration
DEFAULT_WORKOUT_PATH = os.path.join(os.getcwd(), 'input-workouts', 'strength-routine-1', '1-lower-body-squat.json')
INPUT_WORKOUTS_DIR = os.path.join(os.getcwd(), 'input-workouts')
WORKOUT_HISTORY_DIR = os.path.join(os.getcwd(), 'workout-history')
AUDIO_LIB_DIR = os.path.join(os.getcwd(), 'audio-lib')
CLOUD_RUN_URL = os.getenv("CLOUD_RUN_VITPOSE_URL")

# Ensure directories exist
os.makedirs(WORKOUT_HISTORY_DIR, exist_ok=True)
os.makedirs(AUDIO_LIB_DIR, exist_ok=True)

DEBUG = True

class VirtualTrainer:
    def __init__(self):
        self.state = "REST" # REST, ACTIVE
        self.workout_queue = [] # Flattened list of exercises
        self.current_exercise_index = 0
        self.cap = None
        self.model = None
        self.rep_counter = RepCounter()
        self.audio = AudioCoach(AUDIO_LIB_DIR)
        self.validator = PoseValidator(self.audio)
        self.speech_speed = 1.25
        self.pose_history = []
        
        self.audio_loaded = False
        self.audio_queue = []
        self.audio_break_duration = 0
        self.audio_break_start = 0
        self.rest_start_time = 0
        self.start_time = 0
        
        self.setup_logging()
        
        # Clean up old history, keeping the current session
        clean_workout_history(keep_folder=os.path.basename(self.log_dir))
        
        # Hardware Acceleration Check
        self.init_vision_model()
        self.load_workout()

    def setup_logging(self):
        """Sets up logging to a timestamped folder in workout-history."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        # Attempt to derive folder name from workout path
        try:
            path_parts = os.path.normpath(DEFAULT_WORKOUT_PATH).split(os.sep)
            # e.g., strength-routine-1 / 1-lower-body-squat.json
            workout_name = os.path.splitext(path_parts[-1])[0]
            parent_folder = path_parts[-2]
            folder_name = f"{timestamp}-{parent_folder}-{workout_name}"
        except:
            folder_name = f"{timestamp}-workout"
            
        self.log_dir = os.path.join(WORKOUT_HISTORY_DIR, folder_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        logging.basicConfig(
            filename=os.path.join(self.log_dir, "workout.log"),
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        print(f"Logging started in: {self.log_dir}")

    def init_vision_model(self):
        print("Initializing Vision Model...")
        self.model = YOLO("yolo11n-pose.pt")

    def load_workout(self):
        """
        Loads the specific JSON file and flattens the routine into a queue.
        """
        print(f"Loading workout from: {DEFAULT_WORKOUT_PATH}")
        if not os.path.exists(DEFAULT_WORKOUT_PATH):
            print("Error: Workout file not found.")
            return

        with open(DEFAULT_WORKOUT_PATH, 'r') as f:
            data = json.load(f)

        routine = data.get("routine", [])
        self.workout_queue = []

        # Flatten the routine
        for item in routine:
            # Check if it's a group/circuit (has 'activity' as a list)
            if isinstance(item.get("activity"), list):
                # It's a circuit or group
                group_title = item.get("title", "Circuit")
                sets = item.get("sets", 1)
                group_audio = item.get("audio", [])
                if isinstance(group_audio, str): group_audio = [group_audio]
                
                for set_num in range(sets):
                    for i, sub_item in enumerate(item["activity"]):
                        # Create a workout block for each sub-item
                        block = sub_item.copy()
                        block["group_title"] = group_title
                        block["set_current"] = set_num + 1
                        block["set_total"] = sets
                        
                        # Prepend group audio to the first item of the first set
                        if set_num == 0 and i == 0 and group_audio:
                            current_audio = block.get("audio", [])
                            if isinstance(current_audio, str): current_audio = [current_audio]
                            block["audio"] = group_audio + current_audio
                            
                        self.workout_queue.append(block)
                    
                    # Add Rest after circuit set if defined
                    if "rest" in item:
                        self.workout_queue.append({"type": "rest", "duration": item["rest"]})
            else:
                # Single activity
                sets = item.get("sets", 1)
                for set_num in range(sets):
                    block = item.copy()
                    block["set_current"] = set_num + 1
                    block["set_total"] = sets
                    self.workout_queue.append(block)
                    if "rest" in item:
                        self.workout_queue.append({"type": "rest", "duration": item["rest"]})
        
        print(f"Workout Loaded: {len(self.workout_queue)} steps queued.")
        self.preload_audio()

    def preload_audio(self):
        print("Pre-generating audio files...")
        all_texts = set()
        all_texts.add("Done.")

        for item in self.workout_queue:
            audio_list = item.get("audio", [])
            if isinstance(audio_list, str):
                audio_list = [audio_list]
            
            for audio_item in audio_list:
                if isinstance(audio_item, str):
                    all_texts.add(audio_item)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(self.audio.generate_audio, all_texts)
        print("Audio pre-generation complete.")

    def upload_video_async(self, video_path):
        """
        Offloads heavy analysis to Cloud Run during REST periods.
        """
        def _upload():
            if not CLOUD_RUN_URL or "placeholder" in CLOUD_RUN_URL:
                # print("Cloud Run URL not set, skipping upload.")
                return

            print(f"Uploading {video_path} to Cloud Run...")
            try:
                with open(video_path, 'rb') as f:
                    files = {'file': f}
                    response = requests.post(CLOUD_RUN_URL, files=files)
                    if response.status_code == 200:
                        pose_data = response.json()
                        print("Received high-fidelity pose data from Cloud.")
                        # Trigger Gemini analysis here with pose_data
                    else:
                        print(f"Cloud analysis failed: {response.status_code}")
            except Exception as e:
                print(f"Upload error: {e}")

        thread = threading.Thread(target=_upload)
        thread.start()

    def run(self):
        self.cap = cv2.VideoCapture(0)
        frame_count = 0
        while self.cap.isOpened(): # Main Loop
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_count += 1
            self.validator.set_frame_dimensions(frame.shape[0], frame.shape[1])

            if DEBUG and frame_count % 30 == 0: # Print every ~1 sec
                print(f"[DEBUG] State: {self.state} | Idx: {self.current_exercise_index} | AudioQ: {len(self.audio_queue)}")
                logging.debug(f"Loop State: {self.state}, Index: {self.current_exercise_index}")

            if self.current_exercise_index >= len(self.workout_queue):
                cv2.putText(frame, "WORKOUT COMPLETE", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow('Virtual Private Trainer', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # Run Inference
            # Use track() to persist IDs for the validator
            results = list(self.model.track(frame, persist=True, verbose=False, stream=True))

            # Validate and Get Active User
            active_result, active_idx = self.validator.get_active_result(results)
            
            if active_result is None or active_idx is None:
                cv2.putText(frame, "USER NOT DETECTED", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                # Extract keypoints for the active user
                # active_result.keypoints.xy is (N, 17, 2), we want the specific index
                kpts_tensor = active_result.keypoints.xy[active_idx]
                kpts_list = kpts_tensor.cpu().numpy().tolist()
                kpts_numpy = kpts_tensor.cpu().numpy()
                
                # Check framing (Audio feedback)
                self.validator.validate_framing(kpts_numpy)
                
                # Save Pose Data (Only for active user)
                self.pose_history.append({"timestamp": time.time(), "keypoints": kpts_list})

            for result in results:
                # Draw Keypoints
                frame = result.plot()
                
                # Rep Counting Logic
                # Only process if we have an active user and we are in ACTIVE state
                if self.state == "ACTIVE" and active_result is not None:
                    # Get current exercise name
                    current_block = self.workout_queue[self.current_exercise_index]
                    ex_name = current_block.get("activity", "Unknown")
                    
                    # Pass keypoints (xy coordinates)
                    if self.rep_counter.process_pose(kpts_numpy, ex_name):
                        print(f"Rep Counted! Total: {self.rep_counter.reps}")
                        # Optional: Audio feedback for reps
                        # self.audio.speak(str(self.rep_counter.reps))

            # UI Overlay
            status_text = f"State: {self.state}"
            if self.current_exercise_index < len(self.workout_queue):
                item = self.workout_queue[self.current_exercise_index]
                if "activity" in item:
                    target = item.get("reps", item.get("duration", "0s"))
                    status_text += f" | {item['activity']} ({self.rep_counter.reps}/{target})"
                elif item.get("type") == "rest":
                    status_text += f" | REST ({item.get('duration')}s)"

            cv2.putText(frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Virtual Private Trainer', frame)

            # State Machine Logic
            current_block = self.workout_queue[self.current_exercise_index]

            if self.state == "REST":
                # 1. Load Audio
                if not self.audio_loaded:
                    raw_audio = current_block.get("audio", [])
                    if isinstance(raw_audio, str): raw_audio = [raw_audio]
                    self.audio_queue = list(raw_audio)
                    self.audio_loaded = True
                
                # 2. Process Audio
                if pygame.mixer.music.get_busy():
                    pass # Wait for audio to finish
                elif self.audio_break_duration > 0:
                    if time.time() - self.audio_break_start > self.audio_break_duration:
                        self.audio_break_duration = 0
                elif self.audio_queue:
                    next_audio = self.audio_queue.pop(0)
                    if isinstance(next_audio, str):
                        if DEBUG:
                            print(f"[DEBUG] Playing Audio: {next_audio[:30]}...")
                        logging.info(f"Playing Audio: {next_audio}")
                        self.audio.speak(next_audio, speed=self.speech_speed)
                    elif isinstance(next_audio, dict) and "break" in next_audio:
                        self.audio_break_duration = next_audio["break"]
                        self.audio_break_start = time.time()
                else:
                    # Audio finished
                    if current_block.get("type") == "rest":
                        # Handle Rest Duration
                        duration = current_block.get("duration", 30)
                        if self.rest_start_time == 0:
                            self.rest_start_time = time.time()
                        
                        remaining = duration - (time.time() - self.rest_start_time)
                        cv2.putText(frame, f"REST: {int(remaining)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                        
                        if remaining <= 0:
                            logging.info("Rest complete, advancing.")
                            self.current_exercise_index += 1
                            self.audio_loaded = False
                            self.rest_start_time = 0
                    else:
                        # Switch to Active
                        self.state = "ACTIVE"
                        self.start_time = time.time()
                        self.rep_counter.reps = 0
                        logging.info(f"State changed to ACTIVE for {current_block.get('activity')}")

            elif self.state == "ACTIVE":
                current_block = self.workout_queue[self.current_exercise_index]
                target_reps = current_block.get("reps") or current_block.get("reps_per_side")
                target_duration = current_block.get("duration")
                
                is_done = False
                if target_reps and self.rep_counter.reps >= target_reps:
                    is_done = True
                if target_duration and (time.time() - self.start_time) >= target_duration:
                    is_done = True
                
                if is_done:
                    logging.info("Exercise complete.")
                    self.audio.speak("Done.", speed=self.speech_speed)
                    self.state = "REST"
                    self.current_exercise_index += 1
                    self.audio_loaded = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'): # Force next
                self.current_exercise_index += 1
                self.state = "EXERCISE"
                self.rep_counter.reps = 0

        self.save_pose_data()
        self.cap.release()
        cv2.destroyAllWindows()

    def save_pose_data(self):
        if not self.pose_history:
            return
        
        filepath = os.path.join(self.log_dir, "pose_data.json")
        print(f"Saving {len(self.pose_history)} frames of pose data to {filepath}...")
        try:
            with open(filepath, 'w') as f:
                json.dump(self.pose_history, f)
        except Exception as e:
            print(f"Failed to save pose data: {e}")

if __name__ == "__main__":
    trainer = VirtualTrainer()
    trainer.run()
