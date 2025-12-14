from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class SaudiEOUModel:
    def __init__(self, model_name="N0ha/saudi_eou_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def predict_eou_probability(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # label 1 = end-of-utterance
        return probs[0, 1].item()