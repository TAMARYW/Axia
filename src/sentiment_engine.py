import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class SentimentEngine:
    def __init__(self):
        """
        Initialize paths for text data, trained NLP models, and vectorizers.
        """
        self.data_path = os.path.join("data", "news_sentiment.csv")
        self.model_dir = "models"
        self.model_path = os.path.join(self.model_dir, "sentiment_model.pkl")
        self.vectorizer_path = os.path.join(self.model_dir, "sentiment_vectorizer.pkl")
        
        self.model = None
        self.vectorizer = None

    def train_and_save_model(self):
        """
        Loads the financial news dataset, converts text using TF-IDF Vectorization,
        trains a text classification model, and saves both the model and vectorizer.
        """
        print("\n[SentimentEngine] Starting NLP Training Pipeline...")
        
        if not os.path.exists(self.data_path):
            print(f"❌ Error: Sentiment dataset not found at {self.data_path}")
            return False

        # Load the sentiment dataset (Columns: Sentence, Sentiment)
        df = pd.read_csv(self.data_path)
        
        # Drop any missing values just in case
        df = df.dropna(subset=['Sentence', 'Sentiment'])
        
        X = df['Sentence']
        y = df['Sentiment']
        
        # Split into train and test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize TF-IDF Vectorizer to convert text data into numerical features
        print("[SentimentEngine] Converting text data via TF-IDF Vectorization...")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train a Logistic Regression Classifier (Highly effective for TF-IDF text features)
        print("[SentimentEngine] Training Sentiment Classification Model...")
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate internally
        y_pred = self.model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        print(f"📊 [SentimentEngine] Internal Model Accuracy: {acc:.4f}")
        
        # Ensure target directory exists and save both artifacts
        os.makedirs(self.model_dir, exist_ok=True)
        
        with open(self.model_path, 'wb') as f_model:
            pickle.dump(self.model, f_model)
            
        with open(self.vectorizer_path, 'wb') as f_vec:
            pickle.dump(self.vectorizer, f_vec)
            
        print(f"✅ [SentimentEngine] Model and Vectorizer saved successfully.")
        return True

    def load_model(self):
        """
        Loads the pre-trained model and vectorizer from the models directory.
        If missing, automatically runs the training sequence.
        """                        
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            with open(self.model_path, 'rb') as f_model:
                self.model = pickle.load(f_model)
            with open(self.vectorizer_path, 'rb') as f_vec:
                self.vectorizer = pickle.load(f_vec)
            print("✅ [SentimentEngine] Pre-trained NLP models loaded successfully.")
            return True
        else:
            print("⚠️ [SentimentEngine] Pre-trained models missing. Initializing training...")
            return self.train_and_save_model()

    def analyze_text_sentiment(self, text):
        """
        Analyzes the sentiment of a given raw string or list of strings.
        Returns the predicted class (positive, negative, neutral).
        """
        if self.model is None or self.vectorizer is None:
            self.load_model()
            
        # Transform raw user input text using the trained vectorizer
        text_vec = self.vectorizer.transform([text])