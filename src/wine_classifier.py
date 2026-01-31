"""
Wine Quality Classifier
A simple logistic regression model using scikit-learn's wine dataset
"""

import pickle
from pathlib import Path
from typing import Union, Tuple
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class WineClassifier:
    """
    A simple wine quality classifier using Logistic Regression.
    Classifies wines into 3 categories based on chemical properties.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the wine classifier.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.scaler = StandardScaler()
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.is_trained = False
        self.feature_names = None
        self.target_names = None
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the wine dataset and split into train/test sets.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        wine = load_wine()
        self.feature_names = wine.feature_names
        self.target_names = wine.target_names
        
        X_train, X_test, y_train, y_test = train_test_split(
            wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the classifier.
        
        Args:
            X_train: Training features (shape: n_samples, n_features)
            y_train: Training labels (shape: n_samples,)
            
        Raises:
            ValueError: If training data is invalid
        """
        if X_train is None or y_train is None:
            raise ValueError("Training data cannot be None")
        
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty")
        
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict wine class for given samples.
        
        Args:
            X: Feature array (shape: n_samples, n_features)
            
        Returns:
            Array of predicted classes
            
        Raises:
            RuntimeError: If model hasn't been trained
            ValueError: If input is invalid
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        if X is None:
            raise ValueError("Input data cannot be None")
        
        if len(X) == 0:
            raise ValueError("Input data cannot be empty")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for given samples.
        
        Args:
            X: Feature array (shape: n_samples, n_features)
            
        Returns:
            Array of class probabilities (shape: n_samples, n_classes)
            
        Raises:
            RuntimeError: If model hasn't been trained
            ValueError: If input is invalid
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        if X is None:
            raise ValueError("Input data cannot be None")
        
        if len(X) == 0:
            raise ValueError("Input data cannot be empty")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calculate accuracy score on test data.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Accuracy score (0.0 to 1.0)
            
        Raises:
            RuntimeError: If model hasn't been trained
            ValueError: If input is invalid
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before scoring")
        
        if X_test is None or y_test is None:
            raise ValueError("Test data cannot be None")
        
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must have the same length")
        
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.score(X_test_scaled, y_test)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where model should be saved
            
        Raises:
            RuntimeError: If model hasn't been trained
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'scaler': self.scaler,
            'model': self.model,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to saved model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.is_trained = True


def main():
    """Demo function showing basic usage."""
    # Initialize classifier
    classifier = WineClassifier()
    
    # Load and split data
    X_train, X_test, y_train, y_test = classifier.load_data()
    
    # Train model
    print("Training wine classifier...")
    classifier.train(X_train, y_train)
    
    # Evaluate
    train_accuracy = classifier.score(X_train, y_train)
    test_accuracy = classifier.score(X_test, y_test)
    
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    # Make predictions on first 3 test samples
    sample_predictions = classifier.predict(X_test[:3])
    sample_probabilities = classifier.predict_proba(X_test[:3])
    
    print("\nSample predictions:")
    for i, (pred, prob, true) in enumerate(zip(sample_predictions, sample_probabilities, y_test[:3])):
        print(f"  Sample {i+1}: Predicted={pred}, True={true}, Confidence={prob[pred]:.3f}")


if __name__ == "__main__":
    main()