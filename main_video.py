import tkinter as tk
from tkinter import ttk
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import replicate
import os
import requests
import io
import json
import time
from dotenv import load_dotenv
from PIL import Image, ImageTk
import pygame
import cv2

# --- Configuration ---
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_TOKEN:
    print("Error: REPLICATE_API_TOKEN not found. Check your .env file.")

# --- Model Definitions ---
# 1. Transcribe: OpenAI Whisper
MODEL_WHISPER = "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"

# 2. Brain: Llama 3 70B (Restored valid model to prevent 404 errors)
MODEL_BRAIN = "openai/gpt-5-mini"

# 3. Image: Flux Schnell
MODEL_IMAGE = "qwen/qwen-image" #"black-forest-labs/flux-schnell"

# 3.5 Video: Wan Video
MODEL_VIDEO = "wan-video/wan-2.2-i2v-fast"

# 4. Speech: Coqui XTTS v2
MODEL_TTS = "lucataco/xtts-v2:684bc3855b37866c0c65add2ff39c78f3dea3f4ff103a436465326e0f438d55e"

# --- Voice Reference URLs ---
VOICE_MAP = {
    "male": "https://replicate.delivery/pbxt/Jt79w0xsT64R1JsiJ0LQRL8UcWspg5J4RFrU6YwEKpOT1ukS/male.wav",
    "female": "https://audioaiforyou.s3.us-east-2.amazonaws.com/voicemodel/female.wav" 
}
DEFAULT_VOICE = VOICE_MAP["male"]

# --- Audio Recorder Class ---
class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.fs = 44100  # Sample rate

    def start(self):
        self.recording = True
        self.audio_data = []
        self.stream = sd.InputStream(callback=self.callback, channels=1, samplerate=self.fs)
        self.stream.start()

    def stop(self, filename="input_audio.wav"):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        if self.audio_data:
            myrecording = np.concatenate(self.audio_data, axis=0)
            sf.write(filename, myrecording, self.fs)
            return filename
        return None

    def callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

# --- Main Application ---
class HistoryChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Talk to History")
        self.root.geometry("800x900") # Increased height slightly for controls
        self.root.configure(bg="#2c3e50")

        self.is_recording = False
        self.is_paused = False # Flag to track pause state
        self.recorder = AudioRecorder()
        self.generated_images = [] 
        self.audio_file_path = "output_speech.wav"
        self.video_file_path = "output_video.mp4"
        self.current_image_index = 0
        self.canvas_image_ref = None
        self.fade_job = None
        self.video_cap = None

        pygame.mixer.init()
        self.setup_ui()
        self.root.bind('<space>', self.toggle_recording)

    def setup_ui(self):
        # Header
        lbl_title = tk.Label(self.root, text="Ask a Historical Figure", font=("Helvetica", 24, "bold"), bg="#2c3e50", fg="white")
        lbl_title.pack(pady=20)

        # Image Canvas
        self.canvas_size = 512
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="black", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.canvas_text = self.canvas.create_text(256, 256, text="Press SPACE to Record", fill="white", font=("Arial", 16))

        # Status Label
        self.lbl_status = tk.Label(self.root, text="Ready", font=("Arial", 12), bg="#2c3e50", fg="#bdc3c7")
        self.lbl_status.pack(pady=10)

        # Record Button
        self.btn_record = tk.Button(
            self.root, 
            text="Start Recording (Space)", 
            command=self.handle_record_click, 
            font=("Arial", 12, "bold"), 
            width=20, 
            height=2,
            fg="#2d3436",                 
            bg="#81ecec",                 
            highlightbackground="#81ecec" 
        )
        self.btn_record.pack(pady=10)

        # --- UPDATED AUDIO CONTROLS FRAME ---
        self.audio_controls_frame = tk.Frame(self.root, bg="#2c3e50")
        
        # Replay Button
        self.btn_replay = tk.Button(self.audio_controls_frame, text="Replay", command=self.replay_playback, highlightbackground="#2c3e50", width=8)
        self.btn_replay.pack(side=tk.LEFT, padx=5)

        # Pause/Resume Button
        self.btn_play_pause = tk.Button(self.audio_controls_frame, text="Pause", command=self.toggle_playback, highlightbackground="#2c3e50", width=8)
        self.btn_play_pause.pack(side=tk.LEFT, padx=5)

        # Stop Button
        self.btn_stop = tk.Button(self.audio_controls_frame, text="Stop", command=self.stop_playback, highlightbackground="#2c3e50", width=8, fg="red")
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # Volume Slider
        self.vol_slider = ttk.Scale(self.audio_controls_frame, from_=0, to=1, orient=tk.HORIZONTAL, command=self.set_volume)
        self.vol_slider.set(0.8)
        self.vol_slider.pack(side=tk.LEFT, padx=10)
        
        # Label for Volume
        tk.Label(self.audio_controls_frame, text="Vol", bg="#2c3e50", fg="white").pack(side=tk.LEFT)

    # --- Interaction Logic ---
    def toggle_recording(self, event=None):
        # Only allow spacebar to toggle record if we are NOT in playback mode
        if not self.audio_controls_frame.winfo_viewable():
            self.handle_record_click()

    def handle_record_click(self):
        if not self.is_recording:
            self.is_recording = True
            self.btn_record.config(text="Stop Recording (Space)", bg="#fab1a0", highlightbackground="#fab1a0", fg="#2d3436")
            self.lbl_status.config(text="Recording... Speak now.")
            self.recorder.start()
        else:
            self.is_recording = False
            self.btn_record.config(text="Processing...", state=tk.DISABLED, bg="#dfe6e9", highlightbackground="#dfe6e9", fg="#636e72")
            # Unbind space temporarily
            
            filename = self.recorder.stop()
            self.lbl_status.config(text="Processing... (1/4 Transcribing)")
            threading.Thread(target=self.process_pipeline, args=(filename,), daemon=True).start()

    # --- AI Pipeline ---
    def process_pipeline(self, audio_path):
        # 1. Transcribe
        time.sleep(1)
        with open(audio_path, "rb") as file:
            output = replicate.run(MODEL_WHISPER, input={"audio": file})

        # Use actual transcription
        user_text = output.get("transcription") or output.get("text") or str(output)
        print(f"User said: {user_text}")

        # 2. Brain
        self.update_status("Processing... (2/4 Researching Figure)")
        time.sleep(10)
        
        system_prompt = (
            "You are an AI acting as a historical figure. "
            "1. Identify the historical character from the user's input. "
            "2. Determine their gender ('male' or 'female'). "
            "3. Write a dramatic, first-person monologue answering the user. "
            "Output strictly valid JSON: "
            "{\"character_name\": \"Name\", \"gender\": \"male/female\", \"monologue\": \"Text\"} "
            "Do not include markdown."
        )
        
        brain_output = replicate.run(
            MODEL_BRAIN,
            input={
                "prompt": user_text, 
                "system_prompt": system_prompt, 
                "max_tokens": 512,
                "max_new_tokens": 512
            }
        )
        
        full_response = "".join(brain_output)
        clean_json = full_response.replace("```json", "").replace("```", "").strip()
        print("JSON Response:", clean_json)
        
        # try:
        data = json.loads(clean_json)
        figure_name = data.get("character_name")
        gender = data.get("gender").lower()
        monologue = data.get("monologue")
        monologue += "Thank you."
        # except:
        #     figure_name = "Historical Figure"
        #     gender = "male"
        #     monologue = full_response

        print(f"Figure: {figure_name} | Gender: {gender}")

        # 3. Images (Flux)
        self.update_status(f"Processing... (3/4 Painting {figure_name})")
        time.sleep(10)
        
        image_prompt = f"Generate a picture of {figure_name}, hyperrealistic, 8K,looking at directly to the user face to face, speaking, giving a monologue. Should have a microphone standing beside their head and have intense in the eyes like he is talking something very important."
        
        img_output = replicate.run(
            MODEL_IMAGE,
            input={"prompt": image_prompt, "aspect_ratio": "1:1",}# "num_outputs": 1}
        )
        
        self.generated_images = []
        for url in img_output:
            img_data = requests.get(str(url)).content
            img = Image.open(io.BytesIO(img_data))
            img = img.resize((self.canvas_size, self.canvas_size), Image.Resampling.LANCZOS)
            self.generated_images.append(img)

        # 3.5 Video (Wan Video)
        self.update_status(f"Processing... (3.5/4 Animating {figure_name})")
        time.sleep(10)
        
        # Save the generated image to a temporary file
        temp_img_path = "temp_generated_image.png"
        if self.generated_images:
            self.generated_images[0].save(temp_img_path)
            
            video_output = replicate.run(
                MODEL_VIDEO,
                input={
                    "image": open(temp_img_path, "rb"),
                    "prompt": f"A cinematic video of {figure_name} speaking, talking directly to camera, realistic movement", 
                    "aspect_ratio": "1:1"
                }
            )
            
            # Download video
            # video_output might be a list or single item depending on model schema
            video_url = str(video_output[0] if isinstance(video_output, list) else video_output)
            video_data = requests.get(video_url).content
            with open(self.video_file_path, "wb") as f:
                f.write(video_data)
        
        # 4. Speech (XTTS)
        self.update_status("Processing... (4/4 Synthesizing Voice)")
        time.sleep(10)
        
        selected_voice_url = VOICE_MAP.get(gender, DEFAULT_VOICE)
        
        tts_output = replicate.run(
            MODEL_TTS,
            input={
                "text": monologue,
                "language": "en",
                "speaker": selected_voice_url,
                "cleanup_voice": True
            }
        )
        
        with open(self.audio_file_path, "wb") as file:
            file.write(tts_output.read())

        self.root.after(0, self.start_playback)

    def update_status(self, text):
        self.root.after(0, lambda: self.lbl_status.config(text=text))

    # --- Playback Logic & Fading ---
    def start_playback(self):
        self.lbl_status.config(text=f"Listening to response...")
        self.btn_record.pack_forget()
        self.audio_controls_frame.pack(pady=10)
        
        # Reset states
        self.is_paused = False
        self.btn_play_pause.config(text="Pause")
        
        # Get audio duration
        try:
            self.audio_duration = sf.info(self.audio_file_path).duration
        except:
            self.audio_duration = 10 # Fallback

        pygame.mixer.music.load(self.audio_file_path)
        pygame.mixer.music.play()
        
        self.is_fading_out = False
        
        # Initialize Video Capture
        if os.path.exists(self.video_file_path):
            self.video_cap = cv2.VideoCapture(self.video_file_path)
        else:
            self.video_cap = None

        # Create black image for fading
        self.black_img = Image.new("RGB", (self.canvas_size, self.canvas_size), "black")

        self.animate_loop()

    def animate_loop(self):
        # 1. Check if user paused manually. If so, just wait.
        if self.is_paused:
            self.root.after(100, self.animate_loop)
            return

        # 2. Check if audio finished naturally
        if not pygame.mixer.music.get_busy():
            self.lbl_status.config(text="Monologue Finished.")
            self.btn_play_pause.config(text="Finished", state=tk.DISABLED)
            self.btn_stop.config(text="New Chat", bg="#fab1a0", width=12)
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
            return

        # 3. Update Video Frame
        if self.video_cap and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if not ret:
                # Loop video
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_cap.read()
            
            if ret:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((self.canvas_size, self.canvas_size), Image.Resampling.LANCZOS)
                
                # Check for Fade Out
                current_pos_sec = pygame.mixer.music.get_pos() / 1000
                fade_duration = 2.0
                
                if (self.audio_duration - current_pos_sec <= fade_duration):
                    self.is_fading_out = True
                    # Calculate alpha based on time remaining
                    time_left = self.audio_duration - current_pos_sec
                    if time_left < 0: time_left = 0
                    alpha = 1.0 - (time_left / fade_duration) # 0 (full image) -> 1 (black)
                    if alpha > 1.0: alpha = 1.0
                    
                    img = Image.blend(img, self.black_img, alpha)

                self.tk_image = ImageTk.PhotoImage(img)
                self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
                self.canvas_image_ref = self.tk_image

        # Schedule next frame (approx 30 FPS = 33ms)
        self.root.after(33, self.animate_loop)

    def fade_step(self, img1, img2, step, total_steps=20):
        # Deprecated/Unused in video mode but kept for compatibility if needed
        pass

    # --- Controls ---
    def toggle_playback(self):
        if not pygame.mixer.music.get_busy() and not self.is_paused:
            return # Nothing playing

        if self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
            self.btn_play_pause.config(text="Pause")
        else:
            pygame.mixer.music.pause()
            self.is_paused = True
            self.btn_play_pause.config(text="Resume")

    def stop_playback(self):
        pygame.mixer.music.stop()
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.reset_ui()

    def replay_playback(self):
        # Reset UI elements for replay
        self.btn_stop.config(text="Stop", bg="#f0f0f0", width=8, fg="red") # Reset Stop button if it was "New Chat"
        self.btn_play_pause.config(state=tk.NORMAL)
        self.start_playback()

    def set_volume(self, val):
        pygame.mixer.music.set_volume(float(val))

    def reset_ui(self):
        self.audio_controls_frame.pack_forget()
        
        # Restore Record Button
        self.btn_record.config(state=tk.NORMAL, text="Start Recording (Space)", bg="#81ecec", highlightbackground="#81ecec", fg="#2d3436")
        self.btn_record.pack(pady=10)
        
        # Restore Stop Button style (in case it was changed to New Chat)
        self.btn_stop.config(text="Stop", bg="#f0f0f0", width=8, fg="red")
        self.btn_play_pause.config(state=tk.NORMAL, text="Pause")
        self.is_paused = False

        # Clear Canvas
        self.canvas.delete("all")
        self.canvas.create_text(256, 256, text="Press SPACE to Record", fill="white", font=("Arial", 16))
        self.lbl_status.config(text="Finished.")

if __name__ == "__main__":
    root = tk.Tk()
    app = HistoryChatApp(root)
    root.mainloop()