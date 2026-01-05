import os
import numpy as np
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

# Try importing speech recognition for training mode
try:
    import speech_recognition as sr
except ImportError:
    sr = None

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
        self.user_orientation = "unknown"
        self.current_side = 1
        
        self.audio_loaded = False
        self.audio_queue = []
        self.audio_break_duration = 0
        self.audio_break_start = 0
        self.waiting_for_state = None
        self.waiting_for_orientation = None
        self.rest_start_time = 0
        self.start_time = 0
        self.program_start_time = time.time()
        
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

    def listen_for_command(self, prompt_text):
        """Listens for a voice command or falls back to terminal input."""
        print(f"\n[Trainer] {prompt_text}")
        self.audio.speak(prompt_text)
        
        # Wait for audio to finish speaking
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        if sr:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("[Mic] Listening...")
                try:
                    audio = r.listen(source, timeout=5)
                    text = r.recognize_google(audio)
                    print(f"[Mic] Heard: {text}")
                    return text
                except sr.WaitTimeoutError:
                    print("[Mic] Timeout.")
                except sr.UnknownValueError:
                    print("[Mic] Could not understand.")
                except Exception as e:
                    print(f"[Mic] Error: {e}")
        
        # Fallback
        return input(f"[Input] {prompt_text} (Type here): ")

    def learn_exercise(self, exercise_name):
        """Interactive loop to learn a new exercise."""
        logging.info(f"Starting training for {exercise_name}")
        training_data = []
        
        states_to_learn = ["Start Position", "Active Position"] # Default 2 states
        
        for i, default_name in enumerate(states_to_learn):
            # 1. Ask for label
            label = self.listen_for_command(f"Get into the {default_name} and say the name of this state (e.g. 'Up', 'Down').")
            if not label: label = default_name
            
            # 2. Record Frames
            self.audio.speak(f"Hold {label} for 3 seconds.")
            time.sleep(1) # Give time to settle
            
            start_capture = time.time()
            frames_captured = 0
            while time.time() - start_capture < 3.0:
                ret, frame = self.cap.read()
                if not ret: break
                
                results = list(self.model.track(frame, verbose=False, persist=True))
                active_result, active_idx = self.validator.get_active_result(results)
                
                if active_result and active_idx is not None:
                    kpts = active_result.keypoints.xy[active_idx].cpu().numpy()
                    training_data.append((kpts, label))
                    frames_captured += 1
                
            self.audio.speak(f"Captured {frames_captured} frames for {label}.")
            
        self.rep_counter.classifier.train_new_model(exercise_name.lower(), training_data)
        self.audio.speak("Training complete. Resuming workout.")

    def run(self):
        self.cap = cv2.VideoCapture(0)
        frame_count = 0
        while self.cap.isOpened(): # Main Loop
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_count += 1
            self.validator.set_frame_dimensions(frame.shape[0], frame.shape[1])

            if DEBUG:
                if frame_count % 30 == 0: # Print every ~1 sec
                    elapsed = time.time() - self.program_start_time
                    print(f"[DEBUG] Time: {elapsed:.2f}s | Frame: {frame_count} | State: {self.state} | Idx: {self.current_exercise_index} | AudioQ: {len(self.audio_queue)}")
                logging.debug(f"Loop State: {self.state}, Index: {self.current_exercise_index}")

            if self.current_exercise_index >= len(self.workout_queue):
                cv2.putText(frame, "WORKOUT COMPLETE", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                display_frame = cv2.resize(frame, None, fx=2.0, fy=2.0)
                cv2.imshow('Virtual Private Trainer', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # Run Inference
            # Use track() to persist IDs for the validator
            results = list(self.model.track(frame, persist=True, verbose=False, stream=True))

            if DEBUG:
                for r in results:
                    if r.boxes:
                        logging.debug(f"Frame {frame_count} YOLO Detections: {len(r.boxes)}")
                        for i, box in enumerate(r.boxes):
                            b_id = int(box.id[0]) if box.id is not None else None
                            b_cls = int(box.cls[0])
                            b_conf = float(box.conf[0])
                            b_xyxy = [round(x, 1) for x in box.xyxy[0].tolist()]
                            
                            kpts_log = ""
                            if r.keypoints is not None:
                                kpts = r.keypoints.xy[i].cpu().numpy().tolist()
                                kpts_formatted = [[round(p[0], 1), round(p[1], 1)] for p in kpts]
                                kpts_log = f" | Kpts: {kpts_formatted}"

                            logging.debug(f"  ID: {b_id} | Cls: {b_cls} | Conf: {b_conf:.2f} | Box: {b_xyxy}{kpts_log}")

            current_keypoints = None
            rep_counted = False

            # Validate and Get Active User
            active_result, active_idx = self.validator.get_active_result(results)
            
            if active_result is None or active_idx is None:
                cv2.putText(frame, "USER NOT DETECTED", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                self.user_orientation = "unknown"
            else:
                # Extract keypoints for the active user
                # active_result.keypoints.xy is (N, 17, 2), we want the specific index
                kpts_tensor = active_result.keypoints.xy[active_idx]
                kpts_list = kpts_tensor.cpu().numpy().tolist()
                kpts_numpy = kpts_tensor.cpu().numpy()
                current_keypoints = kpts_numpy
                self.user_orientation = self.validator.determine_orientation(current_keypoints)

                # Save Pose Data (Only for active user)
                self.pose_history.append({"timestamp": time.time(), "keypoints": kpts_list})

                # Always process pose to update current_state (even in REST)
                current_block = self.workout_queue[self.current_exercise_index]
                ex_name = current_block.get("activity", "Unknown")
                
                result = self.rep_counter.process_pose(kpts_numpy, ex_name)
                if result == "NEED_TRAINING":
                    # Pause and Learn
                    self.learn_exercise(ex_name)
                else:
                    rep_counted = result

            for result in results:
                # Draw Keypoints
                frame = result.plot()
                
                # Rep Counting Logic
                # Only process if we have an active user and we are in ACTIVE state
                if self.state == "ACTIVE" and active_result is not None:
                    if rep_counted:
                        print(f"Rep Counted! Total: {self.rep_counter.reps}")
                        
                        # Audio Feedback
                        current_block = self.workout_queue[self.current_exercise_index]
                        target_reps = current_block.get("reps") or current_block.get("reps_per_side") or current_block.get("reps_per_side_grouped")
                        if target_reps:
                            remaining = target_reps - self.rep_counter.reps
                            if remaining > 0:
                                self.audio.speak(f"{self.rep_counter.reps}, {remaining} more to go")
                            else:
                                self.audio.speak(f"{self.rep_counter.reps}")

                    # Check Orientation (if activity supports it)
                    if self.rep_counter.current_handler:
                        orient_warning = self.rep_counter.current_handler.check_orientation(current_keypoints, self.user_orientation)
                        self.validator.send_feedback(orient_warning)

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
            
            # Display Exercise State (e.g. Descending, Standing)
            cv2.putText(frame, f"Ex State: {self.rep_counter.current_state.upper()}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

            # Color coding for orientation
            orient_color = (0, 255, 255) # Yellow default
            if self.user_orientation == "front":
                orient_color = (0, 255, 0) # Green
            elif self.user_orientation == "back":
                orient_color = (0, 0, 255) # Red

            orientation_text = f"Facing: {self.user_orientation.capitalize()}"
            cv2.putText(frame, orientation_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, orient_color, 2)

            display_frame = cv2.resize(frame, None, fx=2.0, fy=2.0)
            cv2.imshow('Virtual Private Trainer', display_frame)

            # State Machine Logic
            current_block = self.workout_queue[self.current_exercise_index]

            if self.state == "REST":
                # 1. Load Audio
                if not self.audio_loaded:
                    self.rep_counter.reps = 0
                    raw_audio = current_block.get("audio", [])
                    if isinstance(raw_audio, str): raw_audio = [raw_audio]
                    
                    # Announce rest duration
                    audio_list = list(raw_audio)
                    if current_block.get("type") == "rest":
                        duration = current_block.get("duration", 0)
                        if duration >= 60 and duration % 60 == 0:
                            mins = int(duration / 60)
                            audio_list.append(f"{mins} minute{'s' if mins > 1 else ''} rest")
                        else:
                            audio_list.append(f"{duration} seconds rest")

                    self.audio_queue = audio_list
                    self.audio_loaded = True
                
                # Check for early start (Skip Intro)
                # If user does a valid rep while audio is playing or queued
                if rep_counted and self.validator.validate_framing(current_keypoints, should_speak=False):
                    target_reps = current_block.get("reps") or current_block.get("reps_per_side") or current_block.get("reps_per_side_grouped")
                    if target_reps:
                        logging.info("User performed rep during intro - Skipping explanation.")
                        pygame.mixer.music.stop()
                        self.audio_queue = []
                        self.waiting_for_state = None
                        self.waiting_for_orientation = None
                        self.audio_break_duration = 0
                        
                        self.state = "ACTIVE"
                        self.start_time = time.time()
                        
                        remaining = target_reps - self.rep_counter.reps
                        if remaining < 0: remaining = 0
                        self.audio.speak(f"Skipping explanation, {self.rep_counter.reps}, {remaining} more to go")
                        continue

                # 2. Process Audio
                if pygame.mixer.music.get_busy():
                    pass # Wait for audio to finish
                elif self.waiting_for_orientation:
                    # Wait for user to correct orientation
                    if self.user_orientation == self.waiting_for_orientation:
                        logging.info(f"User corrected orientation to: {self.waiting_for_orientation}")
                        self.waiting_for_orientation = None
                    # Else continue waiting (loop will redraw frame)
                elif self.waiting_for_state:
                    # Wait for user to reach the target state
                    if self.rep_counter.current_state == self.waiting_for_state:
                        logging.info(f"User reached state: {self.waiting_for_state}")
                        self.waiting_for_state = None
                    # Else: continue waiting
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
                    elif isinstance(next_audio, dict):
                        if "break" in next_audio:
                            self.audio_break_duration = next_audio["break"]
                            self.audio_break_start = time.time()
                        
                        # Handle Side Orientation Tags
                        if "side" in next_audio:
                            target_side = next_audio["side"].lower()
                            # Mapping: Left Side Exercise -> Show Left Side -> Face Right
                            expected_orientation = "right" if target_side == "left" else "left"
                            
                            if self.user_orientation != "unknown" and self.user_orientation != expected_orientation:
                                self.audio.speak(f"Turn around, I need to see your {target_side} side.")
                                self.waiting_for_orientation = expected_orientation

                        # Handle Text or State Cues
                        if "text" in next_audio:
                            self.audio.speak(next_audio["text"], speed=self.speech_speed)
                        else:
                            # Fallback: Check for state keys excluding reserved keywords
                            keys = [k for k in next_audio.keys() if k not in ["break", "side", "text"]]
                            if keys:
                                target_state = keys[0]
                                text = next_audio[target_state]
                                self.audio.speak(text, speed=self.speech_speed)
                                self.waiting_for_state = target_state
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
                        if self.validator.validate_framing(current_keypoints, should_speak=True):
                            self.state = "ACTIVE"
                            self.start_time = time.time()
                            self.rep_counter.reps = 0
                            logging.info(f"State changed to ACTIVE for {current_block.get('activity')}")

            elif self.state == "ACTIVE":
                current_block = self.workout_queue[self.current_exercise_index]
                target_reps = current_block.get("reps")
                grouped_reps = current_block.get("reps_per_side_grouped")
                
                if grouped_reps:
                    target_reps = grouped_reps
                elif not target_reps and current_block.get("reps_per_side"):
                    target_reps = current_block.get("reps_per_side") * 2
                
                target_duration = current_block.get("duration")
                
                is_done = False
                if grouped_reps:
                    if self.rep_counter.reps >= grouped_reps:
                        if self.current_side == 1:
                            self.audio.speak("Switch sides.")
                            self.current_side = 2
                            self.rep_counter.reps = 0
                            if self.rep_counter.current_handler: self.rep_counter.current_handler.reset()
                            time.sleep(3) # Brief pause for user to react
                        else:
                            is_done = True
                elif target_reps and self.rep_counter.reps >= target_reps:
                    is_done = True
                
                if target_duration and (time.time() - self.start_time) >= target_duration:
                    is_done = True
                
                if is_done:
                    logging.info("Exercise complete.")
                    self.audio.speak("Done.", speed=self.speech_speed)
                    self.state = "REST"
                    self.current_exercise_index += 1
                    self.current_side = 1
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
