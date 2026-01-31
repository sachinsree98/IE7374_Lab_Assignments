"""
Tests for wine classifier
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wine_classifier import WineClassifier


@pytest.fixture
def classifier():
    """Fixture to create a fresh classifier instance."""
    return WineClassifier()


@pytest.fixture
def trained_classifier():
    """Fixture to create a trained classifier."""
    clf = WineClassifier()
    X_train, _, y_train, _ = clf.load_data()
    clf.train(X_train, y_train)
    return clf


@pytest.fixture
def sample_data():
    """Fixture to load sample data."""
    clf = WineClassifier()
    return clf.load_data()


class TestWineClassifierInitialization:
    """Test classifier initialization."""
    
    def test_initialization(self, classifier):
        """Test that classifier initializes correctly."""
        assert classifier.is_trained == False
        assert classifier.feature_names is None
        assert classifier.target_names is None
    
    def test_random_state(self):
        """Test random state parameter."""
        clf1 = WineClassifier(random_state=42)
        clf2 = WineClassifier(random_state=42)
        assert clf1.model.random_state == clf2.model.random_state


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_data_returns_four_arrays(self, classifier):
        """Test that load_data returns 4 arrays."""
        X_train, X_test, y_train, y_test = classifier.load_data()
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
    
    def test_load_data_shapes(self, classifier):
        """Test data shapes are correct."""
        X_train, X_test, y_train, y_test = classifier.load_data()
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert X_train.shape[1] == 13  # 13 features
    
    def test_load_data_split_ratio(self, classifier):
        """Test train/test split is approximately 80/20."""
        X_train, X_test, y_train, y_test = classifier.load_data()
        total = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total
        assert 0.75 <= train_ratio <= 0.85  # Allow some tolerance
    
    def test_feature_names_loaded(self, classifier):
        """Test that feature names are loaded."""
        classifier.load_data()
        assert classifier.feature_names is not None
        assert len(classifier.feature_names) == 13


class TestTraining:
    """Test model training functionality."""
    
    def test_train_with_valid_data(self, classifier, sample_data):
        """Test training with valid data."""
        X_train, _, y_train, _ = sample_data
        classifier.train(X_train, y_train)
        assert classifier.is_trained == True
    
    def test_train_with_none_data(self, classifier):
        """Test that training with None raises ValueError."""
        with pytest.raises(ValueError, match="Training data cannot be None"):
            classifier.train(None, None)
    
    def test_train_with_empty_data(self, classifier):
        """Test that training with empty arrays raises ValueError."""
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            classifier.train(np.array([]), np.array([]))
    
    def test_train_with_mismatched_lengths(self, classifier, sample_data):
        """Test that mismatched X and y lengths raise ValueError."""
        X_train, _, y_train, _ = sample_data
        with pytest.raises(ValueError, match="must have the same length"):
            classifier.train(X_train, y_train[:-5])


class TestPrediction:
    """Test prediction functionality."""
    
    def test_predict_without_training(self, classifier, sample_data):
        """Test that prediction without training raises RuntimeError."""
        X_train, _, _, _ = sample_data
        with pytest.raises(RuntimeError, match="must be trained"):
            classifier.predict(X_train)
    
    def test_predict_with_valid_data(self, trained_classifier, sample_data):
        """Test prediction with valid data."""
        _, X_test, _, _ = sample_data
        predictions = trained_classifier.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1, 2] for pred in predictions)
    
    def test_predict_with_none(self, trained_classifier):
        """Test that prediction with None raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            trained_classifier.predict(None)
    
    def test_predict_with_empty_array(self, trained_classifier):
        """Test that prediction with empty array raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            trained_classifier.predict(np.array([]))
    
    def test_predict_proba(self, trained_classifier, sample_data):
        """Test probability prediction."""
        _, X_test, _, _ = sample_data
        probas = trained_classifier.predict_proba(X_test)
        assert probas.shape == (len(X_test), 3)  # 3 classes
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1


class TestScoring:
    """Test scoring functionality."""
    
    def test_score_without_training(self, classifier, sample_data):
        """Test that scoring without training raises RuntimeError."""
        _, X_test, _, y_test = sample_data
        with pytest.raises(RuntimeError, match="must be trained"):
            classifier.score(X_test, y_test)
    
    def test_score_with_valid_data(self, trained_classifier, sample_data):
        """Test scoring with valid data."""
        _, X_test, _, y_test = sample_data
        accuracy = trained_classifier.score(X_test, y_test)
        assert 0.0 <= accuracy <= 1.0
        assert accuracy > 0.8  # Wine dataset should achieve good accuracy
    
    def test_score_with_none(self, trained_classifier):
        """Test that scoring with None raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            trained_classifier.score(None, None)
    
    def test_score_with_mismatched_lengths(self, trained_classifier, sample_data):
        """Test that mismatched test data lengths raise ValueError."""
        _, X_test, _, y_test = sample_data
        with pytest.raises(ValueError, match="must have the same length"):
            trained_classifier.score(X_test, y_test[:-5])


class TestModelPersistence:
    """Test model saving and loading."""
    
    def test_save_untrained_model(self, classifier, tmp_path):
        """Test that saving untrained model raises RuntimeError."""
        model_path = tmp_path / "model.pkl"
        with pytest.raises(RuntimeError, match="Cannot save untrained model"):
            classifier.save(model_path)
    
    def test_save_and_load_model(self, trained_classifier, tmp_path, sample_data):
        """Test saving and loading a trained model."""
        model_path = tmp_path / "model.pkl"
        
        # Save model
        trained_classifier.save(model_path)
        assert model_path.exists()
        
        # Load model
        new_classifier = WineClassifier()
        new_classifier.load(model_path)
        assert new_classifier.is_trained == True
        
        # Test predictions match
        _, X_test, _, _ = sample_data
        pred1 = trained_classifier.predict(X_test)
        pred2 = new_classifier.predict(X_test)
        assert np.array_equal(pred1, pred2)
    
    def test_load_nonexistent_model(self, classifier, tmp_path):
        """Test that loading nonexistent model raises FileNotFoundError."""
        model_path = tmp_path / "nonexistent.pkl"
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            classifier.load(model_path)
    
    def test_save_creates_parent_directory(self, trained_classifier, tmp_path):
        """Test that save creates parent directories if needed."""
        model_path = tmp_path / "nested" / "dir" / "model.pkl"
        trained_classifier.save(model_path)
        assert model_path.exists()


class TestModelAccuracy:
    """Test model performance."""
    
    def test_model_achieves_minimum_accuracy(self, trained_classifier, sample_data):
        """Test that model achieves reasonable accuracy."""
        _, X_test, _, y_test = sample_data
        accuracy = trained_classifier.score(X_test, y_test)
        assert accuracy >= 0.85  # Wine dataset should achieve at least 85%