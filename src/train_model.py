"""
Script to train and save the wine classifier model
"""
from pathlib import Path
from wine_classifier import WineClassifier


def train_and_save_model(output_path: str = "models/wine_model.pkl"):
    """Train the wine classifier and save it."""
    print("Initializing classifier...")
    classifier = WineClassifier()
    
    print("Loading wine dataset...")
    X_train, X_test, y_train, y_test = classifier.load_data()
    
    print(f"Training on {len(X_train)} samples...")
    classifier.train(X_train, y_train)
    
    # Evaluate
    train_acc = classifier.score(X_train, y_train)
    test_acc = classifier.score(X_test, y_test)
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    
    # Save model
    print(f"Saving model to {output_path}...")
    classifier.save(output_path)
    print("Model saved successfully!")
    
    return test_acc


if __name__ == "__main__":
    train_and_save_model()