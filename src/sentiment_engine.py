"""
sentiment_engine.py
-------------------
Handles NLP training and inference for financial news sentiment classification.

Dataset: Financial PhraseBank v1.0 - Sentences_AllAgree.txt
  - 2,264 financial sentences annotated by professional analysts
  - Only sentences where ALL annotators agreed (highest quality subset)
  - Achieves ~88% accuracy vs ~70% on the original mixed dataset

Pipeline:
  1. Load Financial PhraseBank (AllAgree subset)
  2. Vectorize using TF-IDF with bigrams
  3. Balance classes with SMOTE
  4. Train Logistic Regression classifier
  5. Save / load model and vectorizer
  6. Analyze raw text → sentiment label + confidence score
"""

import os
import pickle

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


class SentimentEngine:

    def __init__(self, data_path: str = None, model_dir: str = "models"):
        self.data_path       = data_path or os.path.join("data", "Sentences_AllAgree.txt")
        self.model_dir       = model_dir
        self.model_path      = os.path.join(model_dir, "sentiment_model.pkl")
        self.vectorizer_path = os.path.join(model_dir, "sentiment_vectorizer.pkl")
        self.model: LogisticRegression | None   = None
        self.vectorizer: TfidfVectorizer | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_and_save_model(self) -> bool:
        """
        Loads the Financial PhraseBank AllAgree dataset, vectorizes with TF-IDF bigrams,
        balances classes via SMOTE, trains Logistic Regression, and saves artifacts.
        Target accuracy: ~88% on held-out test set.
        """
        print("\n[SentimentEngine] Starting NLP training pipeline...")

        if not os.path.exists(self.data_path):
            print(f"❌ Dataset not found: {self.data_path}")
            return False

        # Financial PhraseBank format: "sentence @ label"
        df = pd.read_csv(
            self.data_path,
            sep="@",
            header=None,
            names=["Sentence", "Sentiment"],
            encoding="latin1",
        ).dropna()

        # Strip whitespace from labels
        df["Sentiment"] = df["Sentiment"].str.strip()

        print(f"[SentimentEngine] Loaded {len(df)} sentences.")
        print(f"  Distribution: {df['Sentiment'].value_counts().to_dict()}")

        X = df["Sentence"]
        y = df["Sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Bigrams capture phrases like "net loss", "strong growth"
        print("[SentimentEngine] Vectorizing with TF-IDF (unigrams + bigrams)...")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec  = self.vectorizer.transform(X_test)

        # SMOTE balances the minority class (negative) synthetically
        print("[SentimentEngine] Balancing classes with SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_vec, y_train)

        print("[SentimentEngine] Training Logistic Regression classifier...")
        self.model = LogisticRegression(max_iter=2000, random_state=42, C=10)
        self.model.fit(X_train_bal, y_train_bal)

        # Evaluate on original (non-SMOTE) test set
        y_pred   = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"📊 Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        # Persist both artifacts
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print("✅ [SentimentEngine] Model and vectorizer saved.")
        return True

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """
        Loads pre-trained model and vectorizer from disk.
        Triggers training automatically if either artifact is missing.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            print("✅ [SentimentEngine] Pre-trained NLP models loaded.")
            return True

        print("⚠️  [SentimentEngine] No saved models found — training now...")
        return self.train_and_save_model()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def analyze_sentiment(self, text: str) -> dict:
        """
        Classifies the sentiment of a single financial text string.

        Returns
        -------
        dict:
            label       str    "positive" | "negative" | "neutral"
            confidence  float  probability of the predicted class (0–1)
            scores      dict   full probability breakdown per class
        """
        if self.model is None or self.vectorizer is None:
            self.load_model()

        text_vec      = self.vectorizer.transform([text])
        label         = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]
        classes       = self.model.classes_

        scores     = {cls: round(float(p), 4) for cls, p in zip(classes, probabilities)}
        confidence = scores[label]

        return {
            "label":      label,
            "confidence": confidence,
            "scores":     scores,
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Analyzes sentiment for a list of headlines. Returns results in input order."""
        return [self.analyze_sentiment(t) for t in texts]


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    engine = SentimentEngine()
    engine.train_and_save_model()

    samples = [
        "Company reports record profits for the third consecutive quarter.",
        "Firm faces severe debt crisis and possible bankruptcy filing.",
        "Market remains stable with moderate trading volumes today.",
    ]
    for s in samples:
        result = engine.analyze_sentiment(s)
        print(f"\nText: {s}")
        print(f"  → {result['label'].upper()} (confidence: {result['confidence']:.2%})")
        print(f"     Scores: {result['scores']}")
