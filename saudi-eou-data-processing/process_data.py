import argparse
import pandas as pd
import random

def generate_eou_samples(df):
    training_data = []

    for text in df['Preprocessed_text']:
        text = text.strip()

        # Positive sample (EOU)
        training_data.append({"text": text, "label": 1})

        # Negative sample (Not EOU)
        words = text.split()
        if len(words) > 2:
            cut_point = random.randint(2, len(words) - 1)
            incomplete_text = " ".join(words[:cut_point])
            training_data.append({"text": incomplete_text, "label": 0})

    return pd.DataFrame(training_data)

def main(input_path, output_path):
    df = pd.read_csv(input_path)

    # Drop missing
    df = df.dropna(subset=['Preprocessed_text'])

    # Word count filtering
    df['word_count'] = df['Preprocessed_text'].astype(str).apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= 2]

    print(f"Rows after cleaning: {len(df)}")

    final_df = generate_eou_samples(df)

    # Shuffle
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    final_df.to_csv(output_path, index=False)

    print("EOU dataset saved.")
    print(final_df['label'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare EOU training data")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")

    args = parser.parse_args()
    main(args.input, args.output)