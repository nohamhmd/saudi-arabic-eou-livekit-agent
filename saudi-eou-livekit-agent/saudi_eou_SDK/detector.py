class SaudiEOUDetector:
    def __init__(self, model, threshold=0.7):
        self.model = model
        self.threshold = threshold
        self.buffer = []

    def add_partial_transcript(self, text: str) -> bool:
        """
        Returns True when EOU is detected
        """
        if not text.strip():
            return False

        self.buffer.append(text)

        joined_text = " ".join(self.buffer)
        prob = self.model.predict_eou_probability(joined_text)

        print(f"[EOU] prob={prob:.2f} text='{joined_text}'")

        if prob >= self.threshold:
            self.buffer.clear()
            return True

        return False