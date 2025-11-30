# Talk to History 🎙️🏛️

**Talk to History** is an interactive Python application that allows you to have voice conversations with historical figures.

The app captures your voice, uses **AI** to identify who you want to talk to, generates a dramatic monologue from their perspective, paints their portrait in real-time, and synthesizes a unique voice for them.

## 🚀 Features

  * **Voice Input:** Press spacebar to ask a question (e.g., *"I want to ask Napoleon why he invaded Russia"*).
  * **AI Orchestration:**
      * **Transcription:** OpenAI Whisper (via Replicate).
      * **Intelligence:** OpenAI GPT-5-mini (via Replicate) for historical roleplay.
      * **Visuals:** Qwen-image or Flux Schnell (via Replicate) for generating evolving historical portraits.
      * **Voice:** Coqui XTTS v2 (via Replicate) for cloning gender-appropriate voices.
  * **Multimedia Interface:** Smooth image cross-fading, audio playback controls (Pause, Resume, Replay), and volume adjustment.

-----

## 🛠️ System Requirements

### 1\. Python

This project requires Python 3.8 or higher. (Assumed installed).

### 2\. FFmpeg (Required for Audio Processing)

This application uses audio libraries that require **FFmpeg** to be installed on your system.

#### 🍏 For macOS Users

If you do not have **Homebrew** installed, you must install it first to get FFmpeg.

**Step A: Install Homebrew**
Open your Terminal and run the following command (copy and paste):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

*Follow the on-screen instructions (you may need to enter your Mac password).*

**Step B: Install FFmpeg**
Once Homebrew is finished, run this command in Terminal:

```bash
brew install ffmpeg
```

#### 🪟 For Windows Users

Open PowerShell or Command Prompt and run:

```powershell
winget install ffmpeg
```

*Alternatively, download the executable from the [official FFmpeg site](https://ffmpeg.org/download.html) and add it to your System PATH.*

#### 🐧 For Linux Users

```bash
sudo apt-get update
sudo apt-get install ffmpeg python3-tk
```

-----

## 📦 Installation

1.  **Clone or Download this repository** to your local machine.

2.  **Install Python Dependencies:**
    Open your terminal/command prompt in the project folder and run:

    ```bash
    pip install replicate python-dotenv sounddevice soundfile numpy pygame pillow requests
    ```

3.  **Set up the API Key:**
    You need a **Replicate API Token** to run the AI models.

    1.  Go to [Replicate API Tokens](https://www.google.com/search?q=https://replicate.com/account/api-tokens).
    2.  Create a file named `.env` in the project folder.
    3.  Add your token inside it like this:
        ```env
        REPLICATE_API_TOKEN=r8_YourTokenGoesHere...
        ```

-----

## 🎮 How to Run

1.  Run the application:

    ```bash
    python main.py
    ```

2.  **To Start:**

      * The window will open showing a black screen.
      * Press the **Spacebar** (or click the "Start Recording" button).
      * **Speak clearly:** *"I want to talk to Cleopatra and ask her about Julius Caesar."*
      * Press **Spacebar** again to stop recording.

3.  **Processing:**

      * Wait for the AI pipeline (Transcription -\> Text Gen -\> Image Gen -\> Voice Gen).
      * The status bar will update as each step completes.

4.  **Playback:**

      * The historical figure will start speaking.
      * Images will cross-fade on the screen.
      * Use the **Pause**, **Stop**, or **Replay** buttons to control the experience.

-----

## ⚠️ Troubleshooting

**Error: `ReplicateError: ... Rate limit exceeded`**

  * **Cause:** You are on the Replicate Free Tier and made requests too quickly.
  * **Fix:** Wait 60 seconds before trying again, or add credit ($5) to your Replicate account for higher limits.

**Error: `ModuleNotFoundError: No module named 'tkinter'`**

  * **Fix:** Tkinter usually comes with Python, but on some Linux distros or Mac versions (pyenv), it might be missing.
      * *Mac (if using brew python):* `brew install python-tk`
      * *Linux:* `sudo apt-get install python3-tk`

**Button text is invisible on Mac**

  * **Fix:** The code handles this by using specific `fg/bg` colors compatible with macOS dark/light mode. If issues persist, ensure your Python installation is up to date.

-----

## 📄 License

This project is for educational purposes. All AI models used are hosted via Replicate.