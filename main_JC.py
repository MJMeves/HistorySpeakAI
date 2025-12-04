import tkinter as tk
from tkinter import ttk
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import replicate
from replicate.exceptions import ReplicateError
import os
import requests
import io
import json
import time
import random
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
MODEL_WHISPER = "openai/whisper:4d50797290df275329f202e48c76360b3f22b08d28c196cbc54600319435f8d2"
MODEL_BRAIN = "meta/meta-llama-3-70b-instruct"
MODEL_IMAGE = "black-forest-labs/flux-schnell"
MODEL_TTS = "lucataco/xtts-v2:684bc3855b37866c0c65add2ff39c78f3dea3f4ff103a436465326e0f438d55e"
MODEL_I2V = "wan-video/wan-2.5-i2v-fast"


# Voice Reference URLs for XTTS-v2
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
        self.stream = None

    def start(self):
        self.recording = True
        self.audio_data = []
        self.stream = sd.InputStream(callback=self.callback, channels=1, samplerate=self.fs)
        self.stream.start()

    def stop(self, filename="input_audio.wav"):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.audio_data:
            myrecording = np.concatenate(self.audio_data, axis=0)
            sf.write(filename, myrecording, self.fs)
            return filename
        return None

    def callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_data.append(indata.copy())


# --- Main Application ---
class HistoryChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Talk to History - Video Edition")
        self.root.geometry("800x950")
        self.root.configure(bg="#2c3e50")

        self.is_recording = False
        self.is_paused = False
        self.recorder = AudioRecorder()
        self.generated_image = None
        self.audio_file_path = "output_speech.wav"
        self.video_file_path = "output_video.mp4"
        self.canvas_size = 512
        self.canvas_image_ref = None

        # Video playback attributes
        self.video_cap = None
        self.is_playing_video = False
        self.video_thread = None

        pygame.mixer.init()
        self.setup_ui()
        self.root.bind("<space>", self.toggle_recording)

    def run_with_retry(self, model, input_data, max_retries=3, step_name="API call"):
        """
        Run a Replicate model with automatic retry on rate limit errors.
        """
        base_wait = 12
        for attempt in range(max_retries):
            try:
                time.sleep(base_wait + random.uniform(0, 4))
                return replicate.run(model, input=input_data)
            except ReplicateError as e:
                error_msg = str(e)
                if "throttled" in error_msg.lower() or "rate limit" in error_msg.lower():
                    wait_time = 20
                    if "resets in" in error_msg:
                        try:
                            import re
                            match = re.search(r"resets in ~?(\d+)s", error_msg)
                            if match:
                                wait_time = int(match.group(1)) + 5
                        except Exception:
                            pass
                    if attempt < max_retries - 1:
                        self.update_status(
                            f"Rate limited. Waiting {wait_time}s... ({step_name})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(
                            f"Rate limit exceeded after {max_retries} attempts ({step_name})"
                        )
                else:
                    if attempt < max_retries - 1:
                        self.update_status(
                            f"Error: {str(e)[:40]}. Retrying... ({step_name})"
                        )
                        time.sleep(8)
                        continue
                    else:
                        raise
            except Exception as e:
                if attempt < max_retries - 1:
                    self.update_status(
                        f"Error: {str(e)[:40]}. Retrying... ({step_name})"
                    )
                    time.sleep(8)
                    continue
                else:
                    raise

    def setup_ui(self):
        # Header
        lbl_title = tk.Label(
            self.root,
            text="Talk to History - Video Edition",
            font=("Helvetica", 24, "bold"),
            bg="#2c3e50",
            fg="white",
        )
        lbl_title.pack(pady=20)

        # Video/Image Canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(pady=10)
        self.canvas_text = self.canvas.create_text(
            256,
            256,
            text="Press SPACE to Record",
            fill="white",
            font=("Arial", 16),
        )

        # Status Label
        self.lbl_status = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 12),
            bg="#2c3e50",
            fg="#bdc3c7",
        )
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
        )
        self.btn_record.pack(pady=10)

        # Audio Controls Frame
        self.audio_controls_frame = tk.Frame(self.root, bg="#2c3e50")

        # Replay Button
        self.btn_replay = tk.Button(
            self.audio_controls_frame,
            text="Replay",
            command=self.replay_playback,
            width=8,
        )
        self.btn_replay.pack(side=tk.LEFT, padx=5)

        # Pause/Resume Button
        self.btn_play_pause = tk.Button(
            self.audio_controls_frame,
            text="Pause",
            command=self.toggle_playback,
            width=8,
        )
        self.btn_play_pause.pack(side=tk.LEFT, padx=5)

        # Stop Button
        self.btn_stop = tk.Button(
            self.audio_controls_frame,
            text="Stop",
            command=self.stop_playback,
            width=8,
            fg="red",
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # Volume Slider
        self.vol_slider = ttk.Scale(
            self.audio_controls_frame,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            command=self.set_volume,
        )
        self.vol_slider.set(0.8)
        self.vol_slider.pack(side=tk.LEFT, padx=10)

        tk.Label(
            self.audio_controls_frame, text="Vol", bg="#2c3e50", fg="white"
        ).pack(side=tk.LEFT)

    # --- Interaction Logic ---
    def toggle_recording(self, event=None):
        if not self.audio_controls_frame.winfo_viewable():
            self.handle_record_click()

    def handle_record_click(self):
        if not self.is_recording:
            self.is_recording = True
            self.btn_record.config(
                text="Stop Recording (Space)",
                bg="#fab1a0",
                fg="#2d3436",
            )
            self.lbl_status.config(text="Recording... Speak now.")
            self.recorder.start()
        else:
            self.is_recording = False
            self.btn_record.config(
                text="Processing...",
                state=tk.DISABLED,
                bg="#dfe6e9",
                fg="#636e72",
            )

            filename = self.recorder.stop()
            self.lbl_status.config(text="Processing... (1/5 Transcribing)")
            threading.Thread(
                target=self.process_pipeline, args=(filename,), daemon=True
            ).start()

    # --- AI Pipeline ---
    def process_pipeline(self, audio_path):
        try:
            # 1. Transcribe
            self.update_status("Processing... (1/5 Transcribing)")
            with open(audio_path, "rb") as file:
                output = self.run_with_retry(
                    MODEL_WHISPER, {"audio": file}, step_name="Transcription"
                )

            user_text = 'I want to talk to King Tut and I want to know what his favorite food is'
            # (
            #     output.get("transcription")
            #     if isinstance(output, dict)
            #     else (output.get("text") if isinstance(output, dict) else str(output))
            # )
            print(f"User said: {user_text}")

            # 2. Brain - Historical figure + accent
            self.update_status("Processing... (2/5 Researching Figure)")

            system_prompt = (
                "You are an AI acting as a historical figure. "
                "1. Identify the historical character from the user's input. "
                "2. Determine their gender ('male' or 'female'). "
                "3. Write a dramatic, first-person monologue (100-200 words) answering the user. "
                "Output strictly valid JSON: "
                '{"character_name": "Name", "gender": "male/female", "monologue": "Text"} '
                "Do not include markdown or code blocks."
            )

            brain_output = self.run_with_retry(
                MODEL_BRAIN,
                {
                    "prompt": user_text,
                    "system_prompt": system_prompt,
                    "max_tokens": 512,
                },
                step_name="Brain Processing",
            )

            full_response = "".join(brain_output) if hasattr(brain_output, '__iter__') and not isinstance(brain_output, str) else str(brain_output)
            clean_json = (
                full_response.replace("```json", "")
                .replace("```", "")
                .strip()
            )
            print("JSON Response:", clean_json)

            data = json.loads(clean_json)
            figure_name = data.get("character_name", "Historical Figure")
            gender = data.get("gender", "male").lower()
            monologue = data.get("monologue", full_response)

            print(f"Figure: {figure_name} | Gender: {gender}")

            # 3. Image Generation (portrait)
            self.update_status(f"Processing... (3/5 Painting {figure_name})")

            image_prompt = (
                f"A cinematic portrait of {figure_name}, hyperrealistic, 8K quality, "
                "facing directly at camera, neutral expression, front-facing, "
                "dramatic lighting, historical period-accurate clothing, "
                "professional studio photograph, clean background"
            )

            img_output = self.run_with_retry(
                MODEL_IMAGE,
                {"prompt": image_prompt, "aspect_ratio": "1:1", "num_outputs": 1},
                step_name="Image Generation",
            )

            img_url = list(img_output)[0] if hasattr(img_output, '__iter__') else img_output
            img_data = requests.get(str(img_url)).content
            img = Image.open(io.BytesIO(img_data))
            img = img.resize(
                (self.canvas_size, self.canvas_size), Image.Resampling.LANCZOS
            )
            self.generated_image = img

            static_image_path = "static_portrait.jpg"
            img.save(static_image_path)

            # 4. Speech Generation with XTTS-v2
            self.update_status(
                f"Processing... (4/5 Synthesizing {gender} voice)"
            )

            selected_voice_url = VOICE_MAP.get(gender, DEFAULT_VOICE)

            tts_output = self.run_with_retry(
                MODEL_TTS,
                {
                    "text": monologue,
                    "language": "en",
                    "speaker": selected_voice_url,
                    "cleanup_voice": True
                },
                step_name="Voice Synthesis",
            )

            # Save audio
            temp_audio_path = "temp_output_speech.wav"
            with open(temp_audio_path, "wb") as file:
                if hasattr(tts_output, "read"):
                    file.write(tts_output.read())
                else:
                    audio_data = requests.get(str(tts_output)).content
                    file.write(audio_data)

            # Clip audio to maximum 29 seconds for WAN video generator
            audio_data, sample_rate = sf.read(temp_audio_path)
            max_duration = 29.0  # seconds
            max_samples = int(max_duration * sample_rate)
            
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
                print(f"Audio clipped from {len(audio_data)/sample_rate:.2f}s to {max_duration}s")
            
            sf.write(self.audio_file_path, audio_data, sample_rate)
            
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

            # 5. Image-to-Video with WAN 2.5 i2v FAST
            self.update_status(
                f"Processing... (5/5 Creating WAN 2.5 talking video)"
            )

            with open(static_image_path, "rb") as img_file, open(
                self.audio_file_path, "rb"
            ) as aud_file:
                video_output = self.run_with_retry(
                    MODEL_I2V,
                    {
                        "image": img_file,
                        "audio": aud_file,
                        "resolution": "720p",
                        "duration": 5,
                        "prompt": (
                            f"A realistic talking portrait of {figure_name}, "
                            "accurate lip-sync to the given audio, cinematic framing."
                        ),
                        "seed": 0,
                    },
                    step_name="Image-to-Video Generation",
                )

            if isinstance(video_output, (list, tuple)):
                video_url = str(video_output[0])
            else:
                video_url = str(video_output)

            video_data = requests.get(video_url).content
            with open(self.video_file_path, "wb") as file:
                file.write(video_data)

            self.root.after(0, self.start_playback)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.update_status(f"Error: {str(e)[:60]}")
            self.root.after(0, self.reset_ui)

    def update_status(self, text):
        self.root.after(0, lambda: self.lbl_status.config(text=text))

    # --- Playback Logic ---
    def start_playback(self):
        self.lbl_status.config(text="Listening to historical monologue...")
        self.btn_record.pack_forget()
        self.audio_controls_frame.pack(pady=10)

        self.is_paused = False
        self.btn_play_pause.config(text="Pause")

        # Load and play audio
        pygame.mixer.music.load(self.audio_file_path)
        pygame.mixer.music.play()

        # Start video playback
        self.is_playing_video = True
        self.video_thread = threading.Thread(
            target=self.play_video, daemon=True
        )
        self.video_thread.start()

        self.animate_loop()

    def play_video(self):
        """Play video frames synchronized with audio."""
        try:
            self.video_cap = cv2.VideoCapture(self.video_file_path)

            if not self.video_cap.isOpened():
                print("Error: Could not open video file")
                if self.generated_image:
                    self.display_image(self.generated_image)
                return

            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 25

            frame_delay = int(1000 / fps)

            while self.is_playing_video:
                if not self.is_paused:
                    ret, frame = self.video_cap.read()

                    if not ret:
                        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(
                        frame_rgb, (self.canvas_size, self.canvas_size)
                    )
                    img = Image.fromarray(frame_resized)
                    self.root.after(0, self.display_image, img)

                time.sleep(frame_delay / 1000.0)

            if self.video_cap:
                self.video_cap.release()

        except Exception as e:
            print(f"Video playback error: {e}")
            if self.generated_image:
                self.root.after(0, self.display_image, self.generated_image)

    def display_image(self, img):
        """Display an image on the canvas."""
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        self.canvas_image_ref = self.tk_image

    def animate_loop(self):
        if self.is_paused:
            self.root.after(100, self.animate_loop)
            return

        if not pygame.mixer.music.get_busy():
            self.is_playing_video = False
            self.lbl_status.config(text="Monologue Finished.")
            self.btn_play_pause.config(text="Finished", state=tk.DISABLED)
            self.btn_stop.config(text="New Chat", bg="#fab1a0", width=12)
            return

        self.root.after(100, self.animate_loop)

    # --- Controls ---
    def toggle_playback(self):
        if not pygame.mixer.music.get_busy() and not self.is_paused:
            return

        if self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
            self.btn_play_pause.config(text="Pause")
        else:
            pygame.mixer.music.pause()
            self.is_paused = True
            self.btn_play_pause.config(text="Resume")

    def stop_playback(self):
        self.is_playing_video = False
        pygame.mixer.music.stop()
        self.reset_ui()

    def replay_playback(self):
        self.is_playing_video = False
        if self.video_cap:
            self.video_cap.release()

        self.btn_stop.config(text="Stop", bg="#f0f0f0", width=8, fg="red")
        self.btn_play_pause.config(state=tk.NORMAL)
        self.start_playback()

    def set_volume(self, val):
        pygame.mixer.music.set_volume(float(val))

    def reset_ui(self):
        self.is_playing_video = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None

        self.audio_controls_frame.pack_forget()

        self.btn_record.config(
            state=tk.NORMAL,
            text="Start Recording (Space)",
            bg="#81ecec",
            fg="#2d3436",
        )
        self.btn_record.pack(pady=10)

        self.btn_stop.config(text="Stop", bg="#f0f0f0", width=8, fg="red")
        self.btn_play_pause.config(state=tk.NORMAL, text="Pause")
        self.is_paused = False

        self.canvas.delete("all")
        self.canvas.create_text(
            256,
            256,
            text="Press SPACE to Record",
            fill="white",
            font=("Arial", 16),
        )
        self.lbl_status.config(text="Ready for next question.")


if __name__ == "__main__":
    root = tk.Tk()
    app = HistoryChatApp(root)
    root.mainloop()