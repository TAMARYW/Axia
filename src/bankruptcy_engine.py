import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class BankruptcyEngine:
    def __init__(self):
        """
        Initialize paths for data, trained models, and setup the model instance.
        """
        # Base directories relative to the root project folder
        self.data_path = os.path.join("data", "bankruptcy_data.csv")
        self.model_dir = "models"
        self.model_path = os.path.join(self.model_dir, "bankruptcy_rf.pkl")
        self.model = None

    def train_and_save_model(self):
        """
        Loads the static bankruptcy dataset, trains a Random Forest Classifier,
        and saves the trained model state to a pickle file.
        """
        print("\n[BankruptcyEngine] Starting Model Training Pipeline...")
        
        # Fallback check for alternative filename configuration
        if not os.path.exists(self.data_path):
            self.data_path = os.path.join("data", "bankruptcy_data.csv")
            
        if not os.path.exists(self.data_path):
            print(f"❌ Error: Bankruptcy dataset not found at {self.data_path}")
            return False

        # Load dataset with special encoding handling for Taiwanese financial metrics
        df = pd.read_csv(self.data_path, encoding='latin1')
        
        # Separate features (X) and target variable (y)
        X = df.drop(columns=['Flag'])
        y = df['Flag']
        
        # Stratified split to preserve class distribution (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize Random Forest with balanced class weights to counter dataset imbalance
        print("[BankruptcyEngine] Fitting Random Forest Classifier...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X_train, y_train)
        
        # Ensure the target directory for saved models exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Persist the trained model using serialization (pickle)
        with open(self.model_path, 'wb') as file:
            pickle.dump(self.model, file)
            
        print(f"✅ [BankruptcyEngine] Model successfully trained and saved to: {self.model_path}")
        return True

    def load_model(self):
        """
        Loads the serialized model from the models directory.
        If it doesn't exist, triggers the training process.
        """
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as file:
                self.model = pickle.load(file)
            print("✅ [BankruptcyEngine] Pre-trained model loaded successfully from file.")
            return True
        else:
            print("⚠️ [BankruptcyEngine] Pre-trained model not found. Initiating automated training...")
            return self.train_and_save_model()

    def predict_resilience(self, company_features):
        """
        Predicts bankruptcy risk for a given company structure.
        Expects a pandas DataFrame or array containing the matching 95 financial metrics.
        """
        if self.model is None:
            self.load_model()
            
        # Perform prediction (0 = Stable, 1 = Bankrupt Risk)
        prediction = self.model.predict(company_features)
        probabilities = self.model.predict_proba(company_features)
        
        return {
            'prediction': int(prediction[0]),
            'stability_probability': float(probabilities[0][0]),
            'risk_probability': float(probabilities[0][1])
        }

if __name__ == "__main__":
    # Internal component test sequence
    engine = BankruptcyEngine()
    engine.train_and_save_model()