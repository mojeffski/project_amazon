from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.utils import resample
from huggingface_hub import login  # Or use your method of authentication
from dotenv import load_dotenv
import pandas as pd
import re
import os

# from dotenv import load_dotenv
import torch
from sklearn.model_selection import train_test_split
from transformers import pipeline
import sqlite3


get_ipython().system("pip install python-dotenv")

# Load environment variables from the .env file
load_dotenv()

# Access the Hugging Face token
hf_token = os.getenv("HF_TOKEN")

# Use the token to authenticate with Hugging Face
login(token=hf_token)

# Importing individual functions from the cleaning.py module
def label_sentiment(rating):
    if rating >= 4:
        return 2  # Positive
    elif rating == 3:
        return 1  # Neutral
    else:
        return 0  # Negative

def truncate_review(review, max_length=512):
    return review[:max_length]

# Function to extract overall accuracy and precision values for each group
def extract_metrics(classification_report):
    # Extract overall accuracy
    accuracy = re.search(r"accuracy\s+([\d.]+)", classification_report)
    overall_accuracy = float(accuracy.group(1)) if accuracy else None

    # Extract precision values for each group
    precision_values = {}
    lines = classification_report.splitlines()

    # Iterate over each line in the classification report
    for line in lines:
        # Identify lines that contain precision values for the groups
        # (negative, positive, neutral, or stars)
        if (
            "negative" in line
            or "positive" in line
            or "neutral" in line
            or "star" in line
        ):
            parts = line.split()

            # Handle multi-word group names like '1 star' or '2 stars'
            if parts[1] == "star" or parts[1] == "stars":
                # Example: '1 star', '2 stars'
                group = parts[0] + " " + parts[1]
                precision = float(parts[2])  # Precision value is at index 2
            else:
                group = parts[0]  # Example: 'negative', 'positive', 'neutral'
                precision = float(parts[1])  # Precision value is at index 1

            # Store the precision value for the group
            precision_values[group] = precision

    return overall_accuracy, precision_values

# Function to store accuracy and precision metrics in the database
def store_metrics_in_db(model_name, dataset_type, overall_accuracy, precision_values):
    with sqlite3.connect("flask_project/data/blogposts.db", timeout=30) as conn:
        cursor = conn.cursor()

        # Create a table if it doesn't exist for storing the metrics
        table_name = f"{model_name}_{dataset_type}_metrics"
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                group_name TEXT,
                precision REAL,
                overall_accuracy REAL
            )
        """
        )
        # Insert overall accuracy
        cursor.execute(
            f"""
            INSERT INTO {table_name} (group_name, overall_accuracy)
            VALUES (?, ?)
        """,
            ("overall", overall_accuracy),
        )

        # Insert precision values for each group
        for group, precision in precision_values.items():
            cursor.execute(
                f"""
                INSERT INTO {table_name} (group_name, precision)
                VALUES (?, ?)
            """,
                (group, precision),
            )

        # Commit the changes
        conn.commit()

# ## Read Dataset
# path for dataset
project_path = "dataset/original/1429_1.csv"
df = pd.read_csv(project_path)

# Do minimal data cleaning
df["labels"] = df["reviews.rating"].apply(label_sentiment)
df["labels_old"] = df["reviews.rating"]
# exclude empty entries
df = df[df["reviews.text"].notnull()]

# Keep only the columns we need (text and sentiment)
df_clean = df[["reviews.text", "labels", "labels_old"]]
# Display the first few rows of the cleaned data
df_clean.rename(columns={"reviews.text": "reviews_text"}, inplace=True)
df_clean = df_clean.dropna()
df_clean


# ### Resampling

# Separate the classes
df_majority = df_clean[df_clean["labels"] == 2]
df_minority_1 = df_clean[df_clean["labels"] == 1]
df_minority_0 = df_clean[df_clean["labels"] == 0]

# Resample minority classes (oversample class 0 and 1)
df_minority_1_upsampled = resample(
    df_minority_1,
    replace=True,  # sample with replacement
    n_samples=1499,  # match desired size
    random_state=42,
)

df_minority_0_upsampled = resample(
    df_minority_0,
    replace=True,  # sample with replacement
    n_samples=1499,  # match desired size
    random_state=42,
)

# Combine the resampled data
df_balanced = pd.concat([df_majority, df_minority_1_upsampled, df_minority_0_upsampled])

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


# Check the new distribution
print("the original distribution:", df_clean["labels_old"].value_counts())
print("the resampled distribution:", df_balanced["labels_old"].value_counts())


# Train (60%), Test (20%), Validation (20%)
train_data_orig, temp_orig = train_test_split(df_clean, test_size=0.4, random_state=42)
test_data_orig, val_data_orig = train_test_split(
    temp_orig, test_size=0.5, random_state=42
)

# Train (60%), Test (20%), Validation (20%)
train_data_balanced, temp_balanced = train_test_split(
    df_balanced, test_size=0.4, random_state=42
)
test_data_balanced, val_data_balanced = train_test_split(
    temp_balanced, test_size=0.5, random_state=42
)

full_reviews_orig = [
    truncate_review(review) for review in test_data_orig["reviews_text"].tolist()
]
full_reviews_balanced = [
    truncate_review(review) for review in test_data_balanced["reviews_text"].tolist()
]

list_full_reviews = {
    "original": [full_reviews_orig, test_data_orig],
    "balanced": [full_reviews_balanced, test_data_balanced],
}


# ## 1. sentiment classifier

#  - Create a model for classification of customers' reviews (the textual content of the reviews) into positive, neutral, or negative.
#  1. create a testing and training dataset


device = 0 if torch.cuda.is_available() else -1


# Mapping for true labels (in case they are numerical in test_data)
true_label_mapping = {
    "roberta_scale": {0: "negative", 1: "neutral", 2: "positive"},
    "amazon_scale": {
        1: "1 star",
        2: "2 stars",
        3: "3 stars",
        4: "4 stars",
        5: "5 stars",
    },
}
# Load pre-trained sentiment analysis pipeline
model_eval_results = {"roberta": {}, "amazon": {}}

sentiment_models = {
    "roberta": {
        "hf_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "num_labels": 3,
    },
    "amazon": {"hf_id": "LiYuan/amazon-review-sentiment-analysis", "num_labels": 5},
}


# Roberta
list_full_reviews = {
    "original": [full_reviews_orig, test_data_orig],
    "balanced": [full_reviews_balanced, test_data_balanced],
}



sentiment_roberta = pipeline(
    "sentiment-analysis", model=sentiment_models["roberta"]["hf_id"], device=device
)
sentiment_amazon = pipeline(
    "sentiment-analysis", model=sentiment_models["amazon"]["hf_id"], device=device
)

# # 2. Classification

classification_results = {
    "roberta": {"original": {}, "balanced": {}},
    "amazon": {"original": {}, "balanced": {}},
}


# ## 2.1 Origianl Dataaset

# ### 2.1.1: Amazon model

results = sentiment_amazon(list_full_reviews["original"][0])
predicted_sentiments = [result["label"] for result in results]

# Map true labels
true_sentiments = (
    list_full_reviews["original"][1]["labels_old"]
    .map(true_label_mapping["amazon_scale"])
    .tolist()
)

classification_matrix = classification_report(
    true_sentiments,
    predicted_sentiments,
    target_names=list(true_label_mapping["amazon_scale"].values()),
)
conf_matrix = confusion_matrix(
    true_sentiments,
    predicted_sentiments,
    labels=list(true_label_mapping["amazon_scale"].values()),
)

# Add results:
classification_results["amazon"]["original"].update(
    {"classification": classification_matrix, "confusion_matrix": conf_matrix}
)
print("done")
print(classification_matrix)


# ### 2.1.2: Roberta model
results = sentiment_roberta(list_full_reviews["original"][0])
predicted_sentiments = [result["label"] for result in results]

# Map true labels
true_sentiments = (
    list_full_reviews["original"][1]["labels"]
    .map(true_label_mapping["roberta_scale"])
    .tolist()
)

classification_matrix = classification_report(
    true_sentiments,
    predicted_sentiments,
    target_names=list(true_label_mapping["roberta_scale"].values()),
)
conf_matrix = confusion_matrix(
    true_sentiments,
    predicted_sentiments,
    labels=list(true_label_mapping["roberta_scale"].values()),
)

# Add results:
classification_results["roberta"]["original"].update(
    {"classification": classification_matrix, "confusion_matrix": conf_matrix}
)
print("done")
print(classification_matrix)

# ## 2.2: Balanced Dataset
# ### 2.2.1: Amazon model

results = sentiment_amazon(list_full_reviews["balanced"][0])
predicted_sentiments = [result["label"] for result in results]

# Map true labels
true_sentiments = (
    list_full_reviews["balanced"][1]["labels_old"]
    .map(true_label_mapping["amazon_scale"])
    .tolist()
)

classification_matrix = classification_report(
    true_sentiments,
    predicted_sentiments,
    target_names=list(true_label_mapping["amazon_scale"].values()),
)
conf_matrix = confusion_matrix(
    true_sentiments,
    predicted_sentiments,
    labels=list(true_label_mapping["amazon_scale"].values()),
)
# Add results:
classification_results["amazon"]["balanced"].update(
    {"classification": classification_matrix, "confusion_matrix": conf_matrix}
)
print("done")
print(classification_matrix)

len(predicted_sentiments)
len(true_sentiments)

results = sentiment_roberta(list_full_reviews["balanced"][0])
predicted_sentiments = [result["label"] for result in results]

# Map true labels
true_sentiments = (
    list_full_reviews["balanced"][1]["labels"]
    .map(true_label_mapping["roberta_scale"])
    .tolist()
)

classification_matrix = classification_report(
    true_sentiments,
    predicted_sentiments,
    target_names=list(true_label_mapping["roberta_scale"].values()),
)
conf_matrix = confusion_matrix(
    true_sentiments,
    predicted_sentiments,
    labels=list(true_label_mapping["roberta_scale"].values()),
)

# Add results:
classification_results["roberta"]["balanced"].update(
    {"classification": classification_matrix, "confusion_matrix": conf_matrix}
)
print("done")
print(classification_matrix)


#

# Extract and store metrics in the database for both models
for model, data in classification_results.items():
    for dataset_type, result in data.items():
        accuracy, precision = extract_metrics(result["classification"])
        store_metrics_in_db(model, dataset_type, accuracy, precision)
        print(f"Stored metrics for Model: {model}, Dataset: {dataset_type}")


# # 3. Fine tuning
# ## new attempt:
# Initialize the tokenizer (example: for roberta-base model)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Tokenizer function for text tokenization


def tokenize_function(examples):
    return tokenizer(
        examples["reviews_text"], padding="max_length", truncation=True, max_length=128
    )

# Apply tokenization to datasets
train_data, val_test_data = train_test_split(
    df_balanced, test_size=0.3, random_state=42
)
val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)

# Create separate validation datasets for roberta and amazon
val_dataset_roberta = val_dataset.map(tokenize_function, batched=True)
val_dataset_amazon = val_dataset.map(tokenize_function, batched=True)

test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for roberta validation dataset
val_dataset_roberta.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

# Set format for amazon validation dataset (use labels_old instead of labels)
val_dataset_amazon = val_dataset_amazon.map(
    lambda examples: {"labels": examples["labels_old"]}
)
val_dataset_amazon.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

# Set format for train and test datasets
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

assert (
    len(set(val_dataset_amazon["labels_old"])) <= 5
), "Amazon model expects 5 classes!"


sentiment_models = {
    "roberta": {
        "hf_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "num_labels": 3,
    },
    "amazon": {"hf_id": "LiYuan/amazon-review-sentiment-analysis", "num_labels": 5},
}

# Load pre-trained tokenizer and model
model_amazon = sentiment_models["roberta"]["hf_id"]
model_roberta = sentiment_models["amazon"]["hf_id"]
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Function to compute evaluation metrics
def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def fine_tune_and_save_model(model_name, train_dataset, val_dataset, num_labels):
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Move the model to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"./logs/{model_name}",
        load_best_model_at_end=True,
        # Use CPU if no CUDA is available
        no_cuda=(not torch.cuda.is_available()),
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(f"./fine_tuned_model/{model_name}")
    tokenizer.save_pretrained(f"./fine_tuned_model/{model_name}")

    print(f"Fine-tuned model {model_name} saved successfully.")
    return trainer  # Return the trainer for further evaluation

    # Return trainer for later evaluation
    return trainer

# Step 3: Fine-tune both models (Roberta and Amazon)
for model_key, model_info in sentiment_models.items():
    model_name = model_info["hf_id"]
    num_labels = model_info["num_labels"]

    # Fine-tune the model and save it
    trainer = fine_tune_and_save_model(
        model_name, train_dataset, val_dataset, num_labels
    )

    # Store the trained models before moving to the next step
    print(f"Model {model_name} has been fine-tuned and saved.")

# Step 2: Evaluate the model and store metrics
def evaluate_and_store_metrics(
    trainer, model_name, dataset, val_dataset, true_labels_mapping
):
    # Evaluate the model on the validation dataset
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)  # Get the predicted labels

    # Use 'labels' for Roberta and 'labels_old' for Amazon as defined in the
    # datasets
    true_labels = val_dataset["labels"].numpy()

    # Generate the classification report for individual class metrics
    report = classification_report(
        true_labels, preds, target_names=list(true_labels_mapping.values())
    )
    print(f"Classification Report for {model_name}:\n{report}")

    # Calculate and print the confusion matrix
    conf_matrix = confusion_matrix(true_labels, preds)
    print(f"Confusion Matrix for {model_name}:\n{conf_matrix}")

    # Store the classification report and confusion matrix in the
    # classification_results dictionary
    classification_results[model_name][dataset].update(
        {"classification": report, "confusion_matrix": conf_matrix}
    )

    # Extract overall accuracy and precision values from the classification
    # report
    overall_accuracy, precision_values = extract_metrics(report)

    # Store metrics (overall accuracy and precision) in the SQLite database
    store_metrics_in_db(model_name, dataset, overall_accuracy, precision_values)

    # You can optionally store or log the confusion matrix somewhere else (or print it out)
    # You might want to extend this part to store the confusion matrix in the
    # database if needed.

os.environ["WANDB_MODE"] = "disabled"
torch.backends.mps.is_available()

device = 0 if torch.cuda.is_available() else -1


# Define paths to your fine-tuned models
roberta_model_path = "./fine_tuned_model/roberta"
amazon_model_path = "./fine_tuned_model/amazon"

# Load pre-trained tokenizer and model for Roberta
tokenizer_roberta = AutoTokenizer.from_pretrained(roberta_model_path)
model_roberta = AutoModelForSequenceClassification.from_pretrained(roberta_model_path)

# Load pre-trained tokenizer and model for Amazon
tokenizer_amazon = AutoTokenizer.from_pretrained(amazon_model_path)
model_amazon = AutoModelForSequenceClassification.from_pretrained(amazon_model_path)

# Now you can evaluate the models using the same Trainer and evaluation
# functions
# Define training arguments (you can reuse the ones used during fine-tuning)
training_args = TrainingArguments(
    per_device_eval_batch_size=64,
    output_dir="./results/",
    logging_dir="./logs/",
    no_cuda=(not torch.cuda.is_available()),  # Use CPU if no CUDA is available
)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Create the Trainer objects for evaluation
trainer_roberta = Trainer(
    model=model_roberta, args=training_args, eval_dataset=val_dataset_roberta
)

trainer_amazon = Trainer(
    model=model_amazon, args=training_args, eval_dataset=val_dataset_amazon
)

# Evaluate the models and store metrics
evaluate_and_store_metrics(
    trainer_roberta,
    "roberta",
    "balanced",
    val_dataset_roberta,
    true_label_mapping["roberta_scale"],
)

# evaluate_and_store_metrics(trainer_amazon, 'amazon', 'balanced', val_dataset_amazon, true_label_mapping['amazon_scale'])

df = df.dropna(subset=["reviews.text"])

# Make sure to run the model on CPU if GPU is not available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_roberta.to(device)

# Function to get predictions from the model in batches


def predict_sentiment_in_batches(texts, model, tokenizer, batch_size=32):
    all_predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
    return all_predictions


# Apply the model in batches to avoid memory issues
df["sentiment"] = predict_sentiment_in_batches(
    df["reviews.text"].tolist(), model_roberta, tokenizer_roberta
)

# Export the DataFrame with the new sentiment column to a CSV file in the
# 'datasets/interim' folder
export_path = "dataset/interim/cleaned_with_sentiment_numeric.csv"
df.to_csv(export_path, index=False)

print(f"Dataset with sentiment column (0, 1, 2) saved to {export_path}")
