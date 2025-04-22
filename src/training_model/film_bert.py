# !pip install transformers datasets scikit-learn -q

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from google.colab import files
csv_path = "/content/drive/MyDrive/movie_metadata.csv"
df = pd.read_csv(csv_path)


df = df[['review_content', 'review_score_clean']].dropna()
df = df.sample(n=14000, random_state=42) 

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(batch):
    return tokenizer(batch['review_content'], padding="max_length", truncation=True, max_length=256)

tokenized = dataset.map(tokenize_function, batched=True)
tokenized = tokenized.rename_column("review_score_clean", "labels")
tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

training_args = TrainingArguments(
    output_dir="./bert-regression",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"]
)

trainer.train()

preds = trainer.predict(tokenized["test"])
pred_scores = preds.predictions.squeeze()
true_scores = preds.label_ids

mae = mean_absolute_error(true_scores, pred_scores)
r2 = r2_score(true_scores, pred_scores)

print(f"MAE: {mae:.2f}")
print(f"R² : {r2:.3f}")

trainer.save_model("/content/drive/MyDrive/models/bert-regression-final")
tokenizer.save_pretrained("/content/drive/MyDrive/models/bert-regression-final")

