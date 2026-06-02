import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_bankruptcy_model():
    print("==================================================")
    print("🤖 FinBotX - Initializing Machine Learning Pipeline")
    print("==================================================")
    
    # Define possible naming variations due to Windows extension handling
    possible_bankruptcy_paths = [
        os.path.join("data", "bankruptcy_data.csv..csv"),
        os.path.join("data", "bankruptcy_data.csv"),
        os.path.join("data", "bankruptcy_data.csv.csv")
    ]
    
    # Check which file variation exists in the data directory
    bankruptcy_path = None
    for path in possible_bankruptcy_paths:
        if os.path.exists(path):
            bankruptcy_path = path
            break
            
    sentiment_path = os.path.join("data", "data.csv")
    
    # 1. Validate data availability
    if not bankruptcy_path:
        print("❌ Error: Missing bankruptcy dataset in the 'data' directory.")
        return
    if not os.path.exists(sentiment_path):
        print(f"❌ Error: Missing sentiment dataset at {sentiment_path}")
        return

    print(f"📊 Active Bankruptcy File: {bankruptcy_path}")
    print(f"📊 Active Sentiment File: {sentiment_path}")

    # 2. Load the Financial Resilience Dataset with encoding handling
    print("\n[1/3] Loading financial metrics...")
    try:
        df = pd.read_csv(bankruptcy_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("⚠️ UTF-8 decoding failed due to special characters, trying 'latin1'...")
        df = pd.read_csv(bankruptcy_path, encoding='latin1')
        
    print(f"✅ Data loaded: {df.shape[0]} companies with {df.shape[1]} features.")
    
    # 3. Preprocessing: Separate features and target label
    # 'Flag' column represents financial stability (1 = Bankrupt, 0 = Stable)
    X = df.drop(columns=['Flag'])
    y = df['Flag']
    
    # Split dataset into training and testing partitions (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📊 Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")
    
    # 4. Model Training: Random Forest Classifier
    print("\n[2/3] Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("✅ Model training completed successfully.")
    
    # 5. Model Evaluation
    print("\n[3/3] Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n================  Model Metrics  ================")
    print(f"Overall Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("==================================================")

if __name__ == "__main__":
    train_bankruptcy_model()