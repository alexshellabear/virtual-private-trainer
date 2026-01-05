import numpy as np
import os
import json
import importlib.util
import inspect
import sys

class BaseActivity:
    """Base class for external activity classes loaded from utils/activity-classes."""
    def __init__(self):
        self.reps = 0
        self.stage = "UP"
        self.current_state = "unknown"
        self.last_angle = 0

    def is_visible(self, point):
        return np.any(point > 0)

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360-angle
        return angle

    def match(self, exercise_name):
        """Return True if this handler supports the exercise."""
        return False

    def process(self, keypoints):
        """Process keypoints. Return True if rep counted."""
        return False
        
    def reset(self):
        self.reps = 0
        self.stage = "UP"
        self.current_state = "unknown"

    def check_orientation(self, keypoints, current_orientation):
        """Returns a feedback string if orientation is incorrect, else None."""
        return None

class RepCounter:
    def __init__(self):
        self.reps = 0
        self.stage = "UP"  # UP or DOWN
        self.current_exercise = ""
        self.current_state = "unknown"
        self.anchor_y = None # For vertical movement tracking
        self.last_angle = 0
        self.classifier = PoseClassifier()
        self.activities = []
        self.current_handler = None
        self.load_activities()

    def load_activities(self):
        classes_dir = os.path.join(os.getcwd(), 'utils', 'activity-classes')
        os.makedirs(classes_dir, exist_ok=True)
        
        for filename in os.listdir(classes_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                filepath = os.path.join(classes_dir, filename)
                module_name = filename[:-3]
                try:
                    spec = importlib.util.spec_from_file_location(module_name, filepath)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and issubclass(obj, BaseActivity) and obj is not BaseActivity:
                                self.activities.append(obj())
                                print(f"Loaded activity handler: {name}")
                except Exception as e:
                    print(f"Failed to load activity {filename}: {e}")

    def process_pose(self, keypoints, exercise_name):
        """
        Heuristic rep counting based on exercise name.
        Keypoints expected: COCO format (17 keypoints).
        """
        # Map COCO Keypoints (Ultralytics default)
        # 0: Nose
        # 1: L-Eye, 2: R-Eye
        # 3: L-Ear, 4: R-Ear
        # 5: L-Shoulder, 6: R-Shoulder
        # 7: L-Elbow, 8: R-Elbow
        # 9: L-Wrist, 10: R-Wrist
        # 11: L-Hip, 12: R-Hip
        # 13: L-Knee, 14: R-Knee
        # 15: L-Ankle, 16: R-Ankle
        
        ex_name = exercise_name.lower()
        self.current_exercise = exercise_name
        
        # 1. Check Dynamic Activities
        for activity in self.activities:
            if activity.match(ex_name):
                if self.current_handler != activity:
                    activity.reset()
                    self.current_handler = activity
                
                # Sync if external reset happened (Trainer sets reps=0)
                if self.reps == 0 and activity.reps != 0:
                    activity.reset()
                
                counted = activity.process(keypoints)
                self.reps = activity.reps
                self.current_state = activity.current_state
                return counted
        
        self.current_handler = None

        # --- Fallback: Use Pose Classification (k-NN) ---
        # If we have a model for this exercise, use it.
        if self.classifier.has_model(ex_name):
            state = self.classifier.predict(keypoints, ex_name)
            if state:
                self.current_state = state
                # Simple Rep Logic for Classifier: Cycle detection
                # We assume a rep is a cycle returning to the first state defined (usually 'start' or 'up')
                # For simplicity, we track change. If we went A -> B -> A, that's a rep.
                
                # Initialize stage if needed
                if self.stage not in [0, 1]: 
                    self.stage = state

                if state != self.stage:
                    # State changed
                    # If we returned to the "primary" state (the first one trained), count rep
                    primary_state = self.classifier.get_primary_state(ex_name)
                    if state == primary_state and self.stage != primary_state:
                        self.reps += 1
                        self.stage = state
                        return True
                    self.stage = state
            return False
        
        # If no model exists, signal the Trainer to enter learning mode
        return "NEED_TRAINING"

class SimpleKNN:
    """Lightweight k-NN implementation to avoid heavy sklearn dependency."""
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        if len(self.X_train) == 0: return None
        # Euclidean distance
        dists = np.linalg.norm(self.X_train - X, axis=1)
        # Get k nearest
        k = min(self.k, len(self.X_train))
        idx = np.argsort(dists)[:k]
        nearest_labels = self.y_train[idx]
        # Majority vote
        unique, counts = np.unique(nearest_labels, return_counts=True)
        return unique[np.argmax(counts)]

class PoseClassifier:
    def __init__(self):
        self.models = {} # {exercise_name: SimpleKNN}
        self.model_data = {} # {exercise_name: {'X': [], 'y': [], 'states': []}}
        self.data_dir = os.path.join(os.getcwd(), 'utils', 'activity-data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.models_file = os.path.join(self.data_dir, 'custom_pose_models.json')
        self.load_models()

    def normalize_keypoints(self, keypoints):
        """
        Normalizes keypoints to be translation and scale invariant.
        Centers around hip center, scales by torso length.
        """
        kpts = np.array(keypoints)
        # Center: Midpoint of hips (11, 12)
        if np.any(kpts[11] > 0) and np.any(kpts[12] > 0):
            center = (kpts[11] + kpts[12]) / 2
        else:
            center = np.mean(kpts[kpts[:, 0] > 0], axis=0) if np.any(kpts > 0) else np.array([0, 0])
            
        kpts_centered = kpts - center
        
        # Scale: Torso length (Mid-Shoulder to Mid-Hip)
        # Shoulders: 5, 6
        if np.any(kpts[5] > 0) and np.any(kpts[6] > 0):
            shoulder_center = (kpts[5] + kpts[6]) / 2
            # Re-calculate center relative to raw kpts for distance
            hip_center = center 
            scale = np.linalg.norm(shoulder_center - hip_center)
        else:
            scale = 100.0 # Default fallback
            
        if scale == 0: scale = 1
        
        return (kpts_centered / scale).flatten()

    def has_model(self, exercise_name):
        return exercise_name in self.models

    def get_primary_state(self, exercise_name):
        if exercise_name in self.model_data:
            states = self.model_data[exercise_name].get('states', [])
            if states: return states[0]
        return None

    def train_new_model(self, exercise_name, data_samples):
        """
        data_samples: list of (keypoints, label_string)
        """
        X = [self.normalize_keypoints(k) for k, l in data_samples]
        y = [l for k, l in data_samples]
        
        knn = SimpleKNN(k=3)
        knn.fit(X, y)
        
        self.models[exercise_name] = knn
        self.model_data[exercise_name] = {
            'X': [x.tolist() for x in X],
            'y': y,
            'states': list(dict.fromkeys(y)) # Preserve order, unique
        }
        self.save_models()

    def predict(self, keypoints, exercise_name):
        if exercise_name not in self.models: return None
        norm_kpts = self.normalize_keypoints(keypoints)
        return self.models[exercise_name].predict(norm_kpts)

    def save_models(self):
        try:
            with open(self.models_file, 'w') as f:
                json.dump(self.model_data, f)
        except Exception as e:
            print(f"Error saving models: {e}")

    def load_models(self):
        if not os.path.exists(self.models_file): return
        try:
            with open(self.models_file, 'r') as f:
                data = json.load(f)
                for name, model_dat in data.items():
                    knn = SimpleKNN(k=3)
                    knn.fit(model_dat['X'], model_dat['y'])
                    self.models[name] = knn
                    self.model_data[name] = model_dat
        except Exception as e:
            print(f"Error loading models: {e}")