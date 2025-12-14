# Saudi Arabic End-of-Utterance (EOU) Detection & LiveKit Agent

This project fine-tunes a lightweight Arabic Transformer to detect **End-of-Utterance (EOU)** in Saudi Arabic conversational text and speech. The model is integrated into a LiveKit agent that handles **real-time voice and text interactions**, including code-switching with English.

## Key Features
- **Dialect-Specific:** Fine-tuned on Saudilang SCC corpus for Saudi Arabic and code-switching.  
- **Real-Time Turn-Taking:** Low-latency EOU detection.  
- **LiveKit Integration:** Full voice/text support.  
- **Voice Capabilities:**  
  - Input: Arabic speech recognition via **Whisper STT**  
  - Output: Saudi dialect responses via TTS  

---
## Saudi EOU SDK (Reusable Module)

The `saudi_eou_SDK` module is a lightweight, reusable End-of-Utterance detection SDK
that can be imported into **any LiveKit agent** or real-time conversational system.

## Setup

**Prerequisites:**  
- Docker (for LiveKit server)  

**Clone repository:**  
```bash
git clone <repo-url>
cd saudi-eou-project
pip install -r requirements.txt

## To run the priject: 

**1. Data Processing**

Navigate to the folder:
cd saudi-eou-data-processing

Run processing script:
python process_data.py --input <path/to/raw_dataset.csv> --output <path/to/preprocessed_dataset.csv>

Processing Steps:

Cleans missing values.

Filters single-word filler segments.

Generates synthetic negative samples (truncated = label 0).

Keeps complete utterances as positive samples (label 1).


**2. Model Fine-Tuning**

Fine-tunes an AraBERT-based Mini-BERT model for binary EOU classification.

Navigate to the folder:
cd ../saudi-eou-training

Run training:
python train_model.py \
    --data <path/to/dataset.csv> \
    --model asafaya/bert-mini-arabic \
    --output outputs/best_saudi_eou_model

Configuration:

Train/Eval split: 80/20

Max sequence length: 128 tokens

Metrics: F1, Precision, Recall, Accuracy

Early stopping: 3 epochs without improvement

Output: Best model and tokenizer saved in outputs.

**3. LiveKit Agent**

Integrates the fine-tuned EOU model into a real-time voice/text agent.

Navigate to the folder:
cd ../saudi-eou-livekit-agent

Start LiveKit server (Docker):

docker run --rm -p 7880:7880 -p 7881:7881 livekit/livekit-server --keys <API_KEY>:<API_SECRET>

Run the agent (console mode):

python livekit_agent.py console

Workflow:

Input: Receives voice (Whisper STT) or text.

Detection: Fine-tuned EOU model identifies end of turn.

Response: Generates Saudi Arabic text.

Output: Reads response aloud (TTS).