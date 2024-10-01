# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# 1. Data Collection and Preprocessing
class MovieReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def load_and_preprocess_data():
    # For this example, we'll create a small synthetic dataset
    # In a real project, you'd load data from a file or API
    reviews = [
        "This movie was fantastic! I loved every minute of it.",
        "Terrible waste of time. Do not watch.",
        "Great acting, but the plot was a bit weak.",
        "One of the best films I've seen this year!",
        "Boring and predictable. Very disappointed."
    ]
    labels = [1, 0, 1, 1, 0]  # 1 for positive, 0 for negative
    
    # Preprocess the reviews
    processed_reviews = [preprocess_text(review) for review in reviews]
    
    return processed_reviews, labels

# 2. Model Training
def train_model(train_dataloader, model, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    return total_loss / len(train_dataloader)

# 3. Model Evaluation
def evaluate_model(eval_dataloader, model, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    return predictions, actual_labels

def calculate_metrics(predictions, actual_labels):
    accuracy = accuracy_score(actual_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 4. Main function to run the entire pipeline
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess data
    texts, labels = load_and_preprocess_data()
    
    # Split data
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=2
    ).to(device)
    
    # Create datasets and dataloaders
    train_dataset = MovieReviewDataset(train_texts, train_labels, tokenizer)
    eval_dataset = MovieReviewDataset(eval_texts, eval_labels, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        avg_loss = train_model(train_dataloader, model, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    # Evaluation
    predictions, actual_labels = evaluate_model(eval_dataloader, model, device)
    metrics = calculate_metrics(predictions, actual_labels)
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # 5. Application to new data
    def predict_sentiment(text):
        processed_text = preprocess_text(text)
        encoded = tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
        
        return "Positive" if preds.item() == 1 else "Negative"
    
    # Example application
    new_review = "I just watched this movie and it was absolutely amazing!"
    sentiment = predict_sentiment(new_review)
    print(f"\nSample Prediction:")
    print(f"Review: {new_review}")
    print(f"Predicted sentiment: {sentiment}")

if __name__ == "__main__":
    main()