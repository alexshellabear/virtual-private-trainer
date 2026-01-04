import os
import json
import hashlib
import time
import pygame
import subprocess
import shutil
from gtts import gTTS

class AudioCoach:
    def __init__(self, audio_lib_dir):
        self.audio_lib_dir = audio_lib_dir
        self.dict_path = os.path.join(self.audio_lib_dir, "text-to-audio-dict.json")
        
        self.audio_map = {}
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'r') as f:
                self.audio_map = json.load(f)
        
        # Initialize Pygame Mixer for audio playback
        pygame.mixer.init()
        pygame.mixer.music.set_volume(1.0)
        
        self.is_speaking = False
        self.ffmpeg_available = shutil.which("ffmpeg") is not None

    def generate_audio(self, text):
        if not text:
            return None

        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Update dictionary if new
        if text_hash not in self.audio_map:
            self.audio_map[text_hash] = {"text": text, "config": "gTTS native"}
            with open(self.dict_path, 'w') as f:
                json.dump(self.audio_map, f, indent=4)
        
        filename = f"{text_hash}.mp3"
        file_path = os.path.join(self.audio_lib_dir, filename)

        # Generate if not exists
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"[Audio] Generating: {text[:30]}...")
            try:
                tts = gTTS(text=text, lang='en', tld='com.au')
                tts.save(file_path)
            except Exception as e:
                print(f"[Audio] Generation failed: {e}")
                return None
        return file_path

    def speak(self, text, speed=1.0):
        """
        Generates audio using gTTS (cached by hash) and plays it.
        """
        file_path = self.generate_audio(text)
        if file_path:
            if speed != 1.0 and self.ffmpeg_available:
                base, ext = os.path.splitext(file_path)
                speed_str = f"{speed:.2f}".replace('.', '_')
                speed_file_path = f"{base}_speed_{speed_str}{ext}"

                if not os.path.exists(speed_file_path):
                    try:
                        cmd = [
                            "ffmpeg", "-y", "-i", file_path,
                            "-filter:a", f"atempo={speed}",
                            "-vn", speed_file_path
                        ]
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        file_path = speed_file_path
                    except Exception as e:
                        print(f"[Audio] Speed adjustment failed: {e}")
                else:
                    file_path = speed_file_path
            self.play_file(file_path)

    def play_file(self, file_path):
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()