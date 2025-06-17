import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


df = pd.read_excel("dataset/MeIA_2025_train.xlsx")
df["labels"] = df["Polarity"]-1

df = df[["Review", "labels"]]
df["labels"] = df["labels"].astype(int)

dataset = Dataset.from_pandas(df)


checkpoint = "distilbert/distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize(batch):
    return tokenizer(batch["Review"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize, batched=True)

dataset = dataset.train_test_split(test_size=0.3)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

train_dataset.set_format("torch")
eval_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=5
)

def custom_eval_metric(y_true, y_pred, labels=[0,1,2,3,4]):
    T_C = len(y_true)
    T_Ci = np.array([np.sum(np.array(y_true) == label) for label in labels])
    _, _, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    weights = 1 - (T_Ci / T_C)
    numerator = np.sum(weights * f1_per_class)
    denominator = np.sum(weights)
    
    return numerator / denominator if denominator != 0 else 0.0

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    eval_p = custom_eval_metric(labels, predictions)

    return {
        "custom_eval_P": eval_p,
        "accuracy": accuracy_score(labels, predictions),
    }



training_args = TrainingArguments(
    output_dir="./sentiment_model",
    eval_strategy="epoch",    
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    logging_dir="./logs",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
