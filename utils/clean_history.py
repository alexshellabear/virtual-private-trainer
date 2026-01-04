import os
import shutil
import argparse

def clean_workout_history(keep_folder=None):
    # Determine paths based on script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    history_dir = os.path.join(root_dir, 'workout-history')

    if not os.path.exists(history_dir):
        print(f"Workout history directory not found at: {history_dir}")
        return

    # Get list of items in history dir
    items = os.listdir(history_dir)
    
    deleted_count = 0
    
    print(f"Cleaning workout history in: {history_dir}")
    if keep_folder:
        print(f"Preserving folder: {keep_folder}")

    for item in items:
        item_path = os.path.join(history_dir, item)
        
        # Only process directories
        if os.path.isdir(item_path):
            if keep_folder and item == keep_folder:
                print(f"Skipping (keeping): {item}")
                continue
            
            try:
                shutil.rmtree(item_path)
                print(f"Deleted: {item}")
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {item}: {e}")

    print(f"Cleanup complete. Deleted {deleted_count} folders.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete workout history folders.")
    parser.add_argument("--keep", type=str, help="The name of the folder to preserve (optional).", default=None)
    
    args = parser.parse_args()
    
    clean_workout_history(args.keep)