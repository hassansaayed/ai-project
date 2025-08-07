#!/usr/bin/env python3
"""
Scikit-learn based spam classifiers
"""

import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

class SklearnSpamClassifier:
    """Base class for scikit-learn based spam classifiers"""
    
    def __init__(self, model_type='naive_bayes'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.model = None
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def create_model(self):
        """Create the appropriate model based on type"""
        if self.model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='linear', random_state=42, probability=True)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, spam_texts, ham_texts):
        """Train the classifier"""
        print(f"Training {self.model_type.replace('_', ' ').title()} classifier...")
        
        # Preprocess all texts
        all_texts = []
        labels = []
        
        # Process spam texts
        for text in spam_texts:
            processed_text = self.preprocess_text(text)
            all_texts.append(processed_text)
            labels.append(1)  # 1 for spam
        
        # Process ham texts
        for text in ham_texts:
            processed_text = self.preprocess_text(text)
            all_texts.append(processed_text)
            labels.append(0)  # 0 for ham
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            all_texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Fit TF-IDF vectorizer
        print("Fitting TF-IDF vectorizer...")
        X_train_vectors = self.vectorizer.fit_transform(X_train)
        X_test_vectors = self.vectorizer.transform(X_test)
        
        # Create and train model
        self.create_model()
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_vectors, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_vectors)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training complete!")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        self.is_trained = True
        return accuracy
    
    def classify_text(self, text):
        """Classify text as spam or ham"""
        if not text.strip():
            return "Please provide some text to classify"
        
        if not self.is_trained:
            return "Model not trained"
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize text
        text_vector = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_vector)[0]
        prediction_label = 'spam' if prediction == 1 else 'ham'
        
        # Get prediction probability
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(text_vector)[0]
            confidence = max(probabilities) * 100
        else:
            # For SVM without probability, use decision function
            decision_score = self.model.decision_function(text_vector)[0]
            confidence = min(95, 50 + abs(decision_score) * 10)
        
        # Calculate spam score for compatibility
        if prediction == 1:  # spam
            spam_score = min(10, confidence / 10)
        else:  # ham
            spam_score = max(0, (100 - confidence) / 10)
        
        # Get feature importance for reasons (if available)
        reasons = []
        if hasattr(self.model, 'feature_importances_'):
            # Random Forest feature importance
            feature_names = list(self.vectorizer.vocabulary_.keys())
            feature_importance = self.model.feature_importances_
            top_features = np.argsort(feature_importance)[-5:]  # Top 5 features
            for idx in reversed(top_features):
                if feature_importance[idx] > 0.01:  # Only significant features
                    reasons.append(f"Important feature: {feature_names[idx]}")
        elif hasattr(self.model, 'coef_'):
            # Logistic Regression coefficients
            feature_names = list(self.vectorizer.vocabulary_.keys())
            coefficients = self.model.coef_[0]
            # Convert to dense array if it's a sparse matrix
            if hasattr(coefficients, 'toarray'):
                coefficients = coefficients.toarray().flatten()
            top_features = np.argsort(coefficients)[-3:]  # Top 3 features
            for idx in reversed(top_features):
                if float(coefficients[idx]) > 0.1:  # Only significant coefficients
                    reasons.append(f"Key indicator: {feature_names[idx]}")
        
        if not reasons:
            reasons = [f"{self.model_type.replace('_', ' ').title()} probability: {prediction_label.upper()} ({confidence:.1f}% confidence)"]
        
        return {
            'prediction': prediction_label,
            'confidence': round(confidence, 2),
            'text': text,
            'spam_score': round(spam_score, 2),
            'reasons': reasons[:3],  # Top 3 reasons
            'algorithm': self.model_type
        }
    
    def save_model(self, filename=None):
        """Save the trained model"""
        if filename is None:
            filename = f'{self.model_type}_sklearn_model.pkl'
        
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename=None):
        """Load a trained model"""
        if filename is None:
            filename = f'{self.model_type}_sklearn_model.pkl'
        
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found")
            return False

# Convenience functions for each model type
def create_naive_bayes_classifier():
    """Create and return a Naive Bayes classifier"""
    return SklearnSpamClassifier('naive_bayes')

def create_logistic_regression_classifier():
    """Create and return a Logistic Regression classifier"""
    return SklearnSpamClassifier('logistic_regression')

def create_svm_classifier():
    """Create and return an SVM classifier"""
    return SklearnSpamClassifier('svm')

def create_random_forest_classifier():
    """Create and return a Random Forest classifier"""
    return SklearnSpamClassifier('random_forest')

# Training data loading function
def load_training_data():
    """Load training data from files"""
    spam_texts = []
    ham_texts = []
    
    # Load spam messages
    if os.path.exists('spam_messages.txt'):
        with open('spam_messages.txt', 'r', encoding='utf-8') as f:
            spam_texts = [line.strip() for line in f if line.strip()]
    
    # Load ham messages
    if os.path.exists('ham_messages.txt'):
        with open('ham_messages.txt', 'r', encoding='utf-8') as f:
            ham_texts = [line.strip() for line in f if line.strip()]
    
    if spam_texts and ham_texts:
        print(f"✅ Loaded {len(spam_texts)} spam messages and {len(ham_texts)} ham messages")
        return spam_texts, ham_texts
    else:
        print("⚠️ No dataset files found, using sample data")
        # Fallback sample data
        spam_texts = [
            "URGENT! You have won a free iPhone! Click here to claim now!",
            "CONGRATULATIONS! You've been selected for a $1000 gift card!",
            "FREE VIAGRA NOW!!! Click here for amazing deals!",
            "Make money fast! Work from home! Earn $5000 per week!",
            "URGENT: Your account has been suspended. Click here to verify!"
        ]
        ham_texts = [
            "Hi, how are you doing? Let's meet for coffee tomorrow.",
            "The meeting is scheduled for 3 PM today. Please bring the reports.",
            "Thanks for your email. I'll get back to you soon.",
            "Can you send me the project files? I need them for the presentation.",
            "Happy birthday! Hope you have a great day!"
        ]
        return spam_texts, ham_texts

if __name__ == "__main__":
    # Test the classifiers
    print("🧪 Testing Scikit-learn Classifiers")
    print("=" * 50)
    
    # Load training data
    spam_texts, ham_texts = load_training_data()
    
    # Test different models
    models = [
        ('naive_bayes', create_naive_bayes_classifier()),
        ('logistic_regression', create_logistic_regression_classifier()),
        ('svm', create_svm_classifier()),
        ('random_forest', create_random_forest_classifier())
    ]
    
    test_texts = [
        "URGENT! You have won a free iPhone!",
        "Hi, how are you doing? Let's meet for coffee.",
        "FREE VIAGRA NOW!!! Click here!",
        "The meeting is scheduled for 3 PM today."
    ]
    
    for model_name, classifier in models:
        print(f"\n📊 Testing {model_name.replace('_', ' ').title()}:")
        
        # Train the model
        accuracy = classifier.train(spam_texts, ham_texts)
        
        # Test predictions
        for text in test_texts:
            result = classifier.classify_text(text)
            print(f"  '{text[:30]}...' -> {result['prediction'].upper()} ({result['confidence']}%)")
        
        # Save the model
        classifier.save_model()
        print()
