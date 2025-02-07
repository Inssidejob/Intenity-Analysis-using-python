# BERT-Based Intensity Analysis
# This script processes text data to classify emotions into Happiness, Anger, and Sadness using BERT.

# Install required libraries (uncomment if not installed)
# !pip install transformers torch pandas scikit-learn

# Step 1: Import Dependenc
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocess

# Step 2: Load and Merge Datasets
happiness_df = pd.read_csv("happiness.csv")
sadness_df = pd.read_csv("sadness.csv")
anger_df = pd.read_csv("anger.csv")

# Assign labels
happiness_df["Label"] = "Happiness"
sadness_df["Label"] = "Sadness"
anger_df["Label"] = "Anger"

# Merge datasets
df = pd.concat([happiness_df, sadness_df, anger_df], ignore_index=True)
df.to_csv("emotion_dataset.csv", index=False)
print("Dataset merged successfully!")

# Step 3: Encode Labels & Split Data
label_encoder = LabelEncoder()
df["encoded_label"] = label_encoder.fit_transform(df["Label"])

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Text"].tolist(), df["encoded_label"].tolist(), test_size=0.2, random_state=42
)

print(f"Training samples: {len(train_texts)}, Testing samples: {len(test_texts)}")

# Step 4: Tokenize Data using BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

print("Tokenization complete!")

# Step 5: Train BERT Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print("BERT model loaded and ready for training!")

# Step 6: Save the Trained Model
model.save_pretrained("bert_intensity_model")
tokenizer.save_pretrained("bert_intensity_model")
print("Model saved successfully!")

# Step 7: Predict on New Text
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    labels = {0: "Happiness", 1: "Anger", 2: "Sadness"}
    return labels[prediction]

# Test Prediction
print(predict_emotion("I am feeling so happy today!"))
