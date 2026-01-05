import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables to ensure GEMINI_API_KEY is available
load_dotenv()

class GeminiClient:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables.")
        else:
            genai.configure(api_key=self.api_key)
        
        self.model = genai.GenerativeModel(model_name)

    def _upload_file(self, path):
        """Uploads a file to the Gemini File API and waits for it to be active."""
        print(f"Uploading file: {path}...")
        file_ref = genai.upload_file(path=path)
        print(f"Upload complete: {file_ref.uri}")

        # Wait for processing (crucial for videos)
        while file_ref.state.name == "PROCESSING":
            print("Processing file...", end="\r")
            time.sleep(2)
            file_ref = genai.get_file(file_ref.name)

        if file_ref.state.name == "FAILED":
            raise ValueError(f"File processing failed for {path}")

        print(f"File is ready: {file_ref.name}")
        return file_ref

    def ask(self, prompt, image=None, video_path=None):
        """
        Queries Gemini with text and optional multimodal data.

        Args:
            prompt (str): The text question or instruction.
            image (PIL.Image or str): Optional PIL Image object or file path.
            video_path (str): Optional path to a video file (mp4, mov, etc.).

        Returns:
            str: The generated response text.
        """
        content = [prompt]

        # 1. Handle Image Input
        if image:
            if isinstance(image, str):
                if os.path.exists(image):
                    content.append(Image.open(image))
                else:
                    print(f"Error: Image path not found: {image}")
            elif isinstance(image, Image.Image):
                content.append(image)

        # 2. Handle Video Input
        if video_path:
            if os.path.exists(video_path):
                video_file = self._upload_file(video_path)
                content.append(video_file)
            else:
                print(f"Error: Video path not found: {video_path}")

        # 3. Generate Content
        try:
            response = self.model.generate_content(content)
            return response.text
        except Exception as e:
            return f"Error calling Gemini API: {e}"

if __name__ == "__main__":
    # Quick test
    client = GeminiClient()
    print(client.ask("Hello! Are you ready to assist with workout analysis?"))