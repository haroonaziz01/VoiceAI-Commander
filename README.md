# 🎙️ VoiceAI Commander

A lightweight AI-powered voice assistant built with Python and Streamlit that recognizes spoken commands and performs system actions — powered by an Artificial Neural Network (ANN) trained on custom intent data.

---

## ✨ Features

- 🎤 **Real-time voice recognition** via Google Speech Recognition API
- 🧠 **ANN-based intent classification** using TensorFlow/Keras
- 🌐 **Browser & web control** — open Google, YouTube, SoundCloud
- 📝 **System actions** — launch Notepad (Windows)
- ⚡ **Instant predictions** using TF-IDF vectorization + Label Encoding
- 🖥️ **Clean Streamlit UI** — simple one-click interface

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| TensorFlow / Keras | ANN model training & inference |
| Scikit-learn | TF-IDF vectorization & label encoding |
| SpeechRecognition | Microphone input & Google STT |
| Streamlit | Web-based UI |
| Pickle | Model artifact serialization |

---

## 📥 Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/voiceai-commander.git
cd voiceai-commander
```

**2. Install dependencies**
```bash
pip install streamlit speechrecognition tensorflow scikit-learn numpy pandas
```

**3. Install PyAudio** (required for microphone access)
```bash
# Windows
pip install pyaudio

# Linux
sudo apt-get install portaudio19-dev
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio
```

---

## 🚀 Usage

**Step 1: Train the model**
```bash
python train.py
```
This generates three files: `model.h5`, `vectorizer.pkl`, and `label_encoder.pkl`.

**Step 2: Launch the assistant**
```bash
streamlit run main.py
```

**Step 3: Click "🎤 Speak Now!"** and say a command.

---

## 🗣️ Supported Commands

| Voice Command (examples) | Action |
|---|---|
| "open browser", "start browser" | Opens Google Chrome |
| "play music", "start music" | Opens SoundCloud |
| "open youtube", "play youtube" | Opens a YouTube playlist |
| "open notepad", "launch notepad" | Opens Notepad *(Windows only)* |
| "shutdown system", "turn off computer" | Disabled for safety |

---

## 📂 Project Structure

```
voiceai-commander/
│
├── main.py              # Streamlit app — voice input + prediction + actions
├── train.py             # ANN training script — generates model artifacts
│
├── model.h5             # Trained Keras model (generated after training)
├── vectorizer.pkl       # TF-IDF vectorizer (generated after training)
└── label_encoder.pkl    # Label encoder (generated after training)
```

---

## ⚙️ How It Works

1. **Training** — `train.py` builds a labeled dataset of voice command phrases, vectorizes them using TF-IDF, and trains a 3-layer ANN (Dense → Dense → Softmax) for multi-class intent classification.
2. **Inference** — When the user clicks "Speak Now!", the app captures microphone audio, converts it to text via Google's STT API, and runs it through the trained model.
3. **Action Dispatch** — The predicted intent label is matched to a `perform_action()` handler that opens URLs or launches system apps.

---

## ⚠️ Known Limitations

| Limitation | Details |
|---|---|
| Small training set | Only 15 samples — accuracy degrades on unseen phrasing |
| Google STT dependency | Requires internet connection for speech recognition |
| Windows-only for Notepad | OS-specific actions not fully cross-platform |
| No wake word | Requires manual button click to activate listening |

---

## 🔮 Planned Improvements

- [ ] Expand training data with more varied command phrases
- [ ] Add offline STT support (Vosk / Whisper)
- [ ] Wake word detection ("Hey Assistant")
- [ ] More system actions — file manager, calculator, custom apps
- [ ] Multi-language support

---

## 👨‍💻 Author

**Haroon Aziz**
Python Developer — AI & Automation

---

## 📄 License

This project is open-source and available for educational and personal use.
