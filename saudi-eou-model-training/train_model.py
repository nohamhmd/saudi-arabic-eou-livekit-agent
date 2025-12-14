import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    # F1-Score is the primary metric for binary classification balance
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
def main(data_path, model_name, output_dir):
    df = pd.read_csv(data_path)

    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch):
       # Padding and Truncation ensure all inputs have a uniform length
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(["text", "__index_level_0__"])
    eval_dataset = eval_dataset.remove_columns(["text", "__index_level_0__"])

    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch",            
        save_strategy="epoch",
        load_best_model_at_end=True,      
        metric_for_best_model='f1',       
        save_total_limit=1,               
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Saudi EOU model")
    parser.add_argument("--data", required=True, help="Prepared CSV file")
    parser.add_argument("--model", default="asafaya/bert-mini-arabic")
    parser.add_argument("--output", default="./outputs/best_saudi_eou_model")

    args = parser.parse_args()

    main(args.data, args.model, args.output)