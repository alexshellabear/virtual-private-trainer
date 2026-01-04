import os
import json
import glob

def delete_cached_audio():
    # Determine the root directory (parent of utils)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    audio_lib_dir = os.path.join(root_dir, 'audio-lib')
    dict_path = os.path.join(audio_lib_dir, 'text-to-audio-dict.json')

    if not os.path.exists(audio_lib_dir):
        print(f"Audio library directory not found at: {audio_lib_dir}")
        return

    # Delete mp3 files
    mp3_files = glob.glob(os.path.join(audio_lib_dir, "*.mp3"))
    print(f"Found {len(mp3_files)} audio files to delete.")
    
    for file_path in mp3_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Failed to delete {os.path.basename(file_path)}: {e}")

    # Reset dictionary
    try:
        with open(dict_path, 'w') as f:
            json.dump({}, f, indent=4)
        print("Reset text-to-audio-dict.json to {}")
    except Exception as e:
        print(f"Failed to reset dictionary: {e}")

if __name__ == "__main__":
    delete_cached_audio()