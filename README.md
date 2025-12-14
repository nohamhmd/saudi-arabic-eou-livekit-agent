# Saudi Arabic End-of-Utterance (EOU) LiveKit Agent

A real-time conversational AI system that detects **End-of-Utterance (EOU)** in **Saudi Arabic**, fine-tuned on a dialect-specific dataset and integrated with **LiveKit** for live voice interactions.

This project includes:

* A data processing pipeline for EOU dataset generation
* Fine-tuning a lightweight Arabic BERT model
* A LiveKit conversational agent with speech, EOU detection, and LLM response generation
* A reusable SDK for EOU inference

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/nohamhmd/saudi-arabic-eou-livekit-agent.git
cd saudi-arabic-eou-livekit-agent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Data Processing

Generate the EOU dataset from the Saudilang Code-Switch Corpus:

```bash
python saudi-eou-data-processing/process_data.py \
  --input <path/to/Saudilang.csv> \
  --output train_eou_saudi_prepared.csv
```

### Dataset Creation Logic

* Clean Saudi Arabic conversational text
* Remove fillers and noise tokens
* Generate **positive samples** (complete utterances)
* Generate **negative samples** (randomly truncated utterances)
* Final dataset is balanced for binary classification

---

## 🧠 Model Fine-Tuning

Train the EOU classifier:

```bash
python saudi-eou-model-training/train_model.py \
  --data train_eou_saudi_prepared.csv \
  --model asafaya/bert-mini-arabic \
  --output ./outputs/best_saudi_eou_model
```
---

## 🎙️ LiveKit Conversational Agent

The fine-tuned EOU model is integrated into a **LiveKit agent** for real-time conversations.

### Prerequisites

* LiveKit server
* OpenAI API key (for Whisper + LLM)
* LiveKit API key & secret:
    If deploying on the cloud, generate them via the cloud dashboard.
  
    If running a local server, generate them with Docker:
    ```bash
    docker run --rm livekit/livekit-server generate-keys
    ```

### Start LiveKit Server

```bash
docker run --rm \
  -p 7880:7880 \
  -p 7881:7881 \
  livekit/livekit-server \
  --keys <YOUR_API_KEY>:<YOUR_API_SECRET>
```

---

### Run the Agent (Console Mode)

```bash
python saudi-eou-livekit-agent/livekit_agent.py console
```

---

## 🗣️ Speech & Conversation Pipeline

1. **LiveKit** streams audio in real time
2. **OpenAI Whisper** performs incremental speech-to-text
3. After each partial transcription, the **EOU model** predicts whether the user has finished speaking
4. Once EOU is detected:

   * The text is sent to an LLM (OpenAI)
   * Instructions enforce friendly Saudi dialect responses
5. **Text-to-Speech (TTS)** outputs audio replies

---

## 📦 EOU SDK Usage

The project includes a reusable SDK for EOU inference:

```python
from saudi_eou_SDK.detector import EOUPredictor
```

---

## 🧩 Design Choices

* **Mini-BERT** for low-latency inference
* Synthetic EOU labeling to simulate real-time speech behavior
* Modular architecture (speech, model, LLM are swappable)
* Saudi dialect focus with English code-switching support
---
