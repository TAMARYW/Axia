"""
bankruptcy_engine.py
--------------------
Handles all machine learning logic for financial resilience prediction.

Pipeline:
  1. Load static bankruptcy dataset and apply column name mapping
  2. Train a Random Forest Classifier with balanced class weights
  3. Save / load the trained model using pickle serialization
  4. Predict bankruptcy risk for a given company's financial profile
  5. Generate SHAP explanations to identify the key risk drivers
"""

import os
import pickle

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.column_mapping import ALL_FEATURES, COLUMN_MAPPING, CRITICAL_FEATURES


class BankruptcyEngine:

    def __init__(self, data_path: str = None, model_dir: str = "models"):
        self.data_path  = data_path or os.path.join("data", "bankruptcy_data.csv")
        self.model_dir  = model_dir
        self.model_path = os.path.join(model_dir, "bankruptcy_rf.pkl")
        self.model: RandomForestClassifier | None = None
        self.feature_names: list[str] = ALL_FEATURES  # 95 clean English names

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_and_save_model(self) -> bool:
        """
        Loads the raw dataset, applies column mapping, trains a Random Forest,
        evaluates it, and persists the model to disk.
        """
        print("\n[BankruptcyEngine] Starting training pipeline...")

        if not os.path.exists(self.data_path):
            print(f"❌ Dataset not found: {self.data_path}")
            return False

        # Load with latin1 encoding (Taiwanese source data)
        df = pd.read_csv(self.data_path, encoding="latin1")

        # Rename all columns to clean English names
        df.rename(columns=COLUMN_MAPPING, inplace=True)

        # Separate features and target
        X = df[ALL_FEATURES]
        y = df["bankruptcy_flag"]

        # Stratified 80/20 split to preserve class imbalance ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("[BankruptcyEngine] Training Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",   # compensates for heavy class imbalance (97% stable)
        )
        self.model.fit(X_train, y_train)

        # Evaluate on held-out test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"📊 Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Stable", "At Risk"]))

        # Persist model
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"✅ Model saved to: {self.model_path}")
        return True

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """
        Loads the pre-trained model from disk.
        If no saved model exists, triggers training automatically.
        """
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            print("✅ [BankruptcyEngine] Pre-trained model loaded.")
            return True

        print("⚠️  [BankruptcyEngine] No saved model found — training now...")
        return self.train_and_save_model()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, company_data: pd.DataFrame) -> dict:
        """
        Predicts bankruptcy risk for a single company.

        Parameters
        ----------
        company_data : pd.DataFrame
            One row with columns matching ALL_FEATURES (95 clean English names).
            Missing optional features are filled with the column median from training data.

        Returns
        -------
        dict with keys:
            prediction          int   0 = Stable, 1 = At Risk
            stability_score     float probability of being stable  (0–1)
            risk_score          float probability of bankruptcy risk (0–1)
            risk_label          str   "Low" / "Medium" / "High"
            top_risk_factors    list  top SHAP-driven features pushing toward risk
        """
        if self.model is None:
            self.load_model()

        # Align columns to training order; fill missing with 0
        input_df = company_data.reindex(columns=self.feature_names, fill_value=0)

        prediction    = int(self.model.predict(input_df)[0])
        probabilities = self.model.predict_proba(input_df)[0]
        risk_score    = float(probabilities[1])
        stable_score  = float(probabilities[0])

        risk_label = (
            "Low"    if risk_score < 0.35 else
            "Medium" if risk_score < 0.65 else
            "High"
        )

        top_factors = self._explain_prediction(input_df)

        return {
            "prediction":       prediction,
            "stability_score":  round(stable_score, 4),
            "risk_score":       round(risk_score, 4),
            "risk_label":       risk_label,
            "top_risk_factors": top_factors,
        }

    # ------------------------------------------------------------------
    # SHAP Explanation
    # ------------------------------------------------------------------

    def _explain_prediction(self, input_df: pd.DataFrame, top_n: int = 10) -> list[dict]:
        """
        Uses SHAP TreeExplainer to identify the top features driving the risk prediction.

        Returns a list of dicts sorted by absolute SHAP impact (descending):
            [{"feature": str, "value": float, "shap_impact": float}, ...]
        """
        explainer   = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(input_df)

        # Handle both old (list) and new (3D array) SHAP output formats
        if isinstance(shap_values, list):
            risk_shap = shap_values[1][0]
        else:
            risk_shap = shap_values[:, :, 1][0]

        factors = []
        for feature, shap_val, raw_val in zip(
            self.feature_names, risk_shap, input_df.values[0]
        ):
            factors.append({
                "feature":     feature,
                "value":       round(float(raw_val), 6),
                "shap_impact": round(float(shap_val), 6),
            })

        # Sort by absolute impact, return top N
        factors.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
        return factors[:top_n]

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def validate_input(self, company_data: dict) -> tuple[bool, list[str]]:
        """
        Checks that all critical features are present and non-null.

        Returns
        -------
        (is_valid: bool, missing_fields: list[str])
        """
        missing = [f for f in CRITICAL_FEATURES if f not in company_data or company_data[f] is None]
        return (len(missing) == 0, missing)


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    engine = BankruptcyEngine()
    engine.train_and_save_model()
