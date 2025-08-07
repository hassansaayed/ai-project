from flask import Flask, render_template, request, jsonify
import re
import os
from sklearn_classifiers import (
    create_naive_bayes_classifier,
    create_logistic_regression_classifier,
    create_svm_classifier,
    create_random_forest_classifier,
    load_training_data
)

app = Flask(__name__)

# Global classifiers
rule_based_classifier = None
naive_bayes_classifier = None
logistic_regression_classifier = None
svm_classifier = None
random_forest_classifier = None

# ============================================================================
# RULE-BASED CLASSIFIER (Keep the original for comparison)
# ============================================================================

class RuleBasedClassifier:
    def __init__(self):
        self.SPAM_KEYWORDS = [
            'urgent', 'free', 'winner', 'congratulations', 'claim', 'limited time',
            'click here', 'earn money', 'work from home', 'make money fast',
            'casino', 'lottery', 'credit card', 'debt', 'loan',
            'investment', 'bitcoin', 'crypto', 'millionaire', 'rich',
            'weight loss', 'diet', 'supplement', 'prescription', 'medication', 'pharmacy', 'discount', 'offer',
            'sale', 'buy now', 'order now', 'limited offer', 'act now',
            'don\'t miss', 'once in a lifetime', 'exclusive', 'secret',
            'guaranteed', 'risk free', 'no risk', 'money back', 'refund'
        ]
        
        self.SPAM_PATTERNS = [
            r'\$[0-9,]+',  # Dollar amounts
            r'[A-Z]{3,}',  # ALL CAPS words
            r'!{2,}',      # Multiple exclamation marks
            r'click here',  # Click here phrases
            r'limited time', # Limited time offers
            r'act now',     # Urgency phrases
            r'winner',      # Winner announcements
            r'free.*[a-z]', # Free + word
            r'earn.*money', # Earn money phrases
            r'work.*home',  # Work from home
        ]
    
    def calculate_spam_score(self, text):
        """Calculate spam score based on keywords and patterns"""
        text_lower = text.lower()
        score = 0
        reasons = []
        
        # Check for spam keywords
        for keyword in self.SPAM_KEYWORDS:
            if keyword in text_lower:
                score += 2
                reasons.append(f"Contains spam keyword: '{keyword}'")
        
        # Check for spam patterns
        for pattern in self.SPAM_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                score += len(matches) * 1.5
                reasons.append(f"Matches spam pattern: '{pattern}' ({len(matches)} matches)")
        
        # Check for excessive capitalization
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:
            score += 3
            reasons.append(f"Excessive capitalization ({caps_ratio:.1%})")
        
        # Check for excessive punctuation
        exclamation_count = text.count('!')
        if exclamation_count > 2:
            score += exclamation_count
            reasons.append(f"Excessive exclamation marks ({exclamation_count})")
        
        # Check for urgency words
        urgency_words = ['urgent', 'immediate', 'now', 'today', 'limited']
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        if urgency_count > 1:
            score += urgency_count
            reasons.append(f"Multiple urgency words ({urgency_count})")
        
        # Check for suspicious links or phone numbers
        if re.search(r'http[s]?://', text):
            score += 2
            reasons.append("Contains URL")
        
        if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text):
            score += 1
            reasons.append("Contains phone number")
        
        return score, reasons
    
    def classify_text(self, text):
        """Classify text as spam or ham"""
        if not text.strip():
            return "Please provide some text to classify"
        
        # Calculate spam score
        score, reasons = self.calculate_spam_score(text)
        
        # Determine classification based on score
        if score >= 8:
            prediction = 'spam'
            confidence = min(95, 70 + score * 2)
        elif score >= 4:
            prediction = 'spam'
            confidence = 60 + score * 3
        elif score >= 2:
            prediction = 'ham'
            confidence = 70 - score * 5
        else:
            prediction = 'ham'
            confidence = 85 - score * 2
        
        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'text': text,
            'spam_score': score,
            'reasons': reasons[:3],
            'algorithm': 'rule_based'
        }

# ============================================================================
# FLASK ROUTES
# ============================================================================

def load_or_create_models():
    """Load existing models or create new ones"""
    global rule_based_classifier, naive_bayes_classifier, logistic_regression_classifier, svm_classifier, random_forest_classifier
    
    # Initialize rule-based classifier
    rule_based_classifier = RuleBasedClassifier()
    print("✅ Rule-based classifier ready")
    
    # Initialize scikit-learn classifiers
    naive_bayes_classifier = create_naive_bayes_classifier()
    logistic_regression_classifier = create_logistic_regression_classifier()
    svm_classifier = create_svm_classifier()
    random_forest_classifier = create_random_forest_classifier()
    
    # Try to load existing models
    models_loaded = 0
    total_models = 4
    
    if naive_bayes_classifier.load_model():
        models_loaded += 1
    else:
        print("Training new Naive Bayes model...")
        spam_data, ham_data = load_training_data()
        naive_bayes_classifier.train(spam_data, ham_data)
        naive_bayes_classifier.save_model()
    
    if logistic_regression_classifier.load_model():
        models_loaded += 1
    else:
        print("Training new Logistic Regression model...")
        spam_data, ham_data = load_training_data()
        logistic_regression_classifier.train(spam_data, ham_data)
        logistic_regression_classifier.save_model()
    
    if svm_classifier.load_model():
        models_loaded += 1
    else:
        print("Training new SVM model...")
        spam_data, ham_data = load_training_data()
        svm_classifier.train(spam_data, ham_data)
        svm_classifier.save_model()
    
    if random_forest_classifier.load_model():
        models_loaded += 1
    else:
        print("Training new Random Forest model...")
        spam_data, ham_data = load_training_data()
        random_forest_classifier.train(spam_data, ham_data)
        random_forest_classifier.save_model()
    
    print(f"✅ {models_loaded}/{total_models} scikit-learn models loaded")

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """API endpoint for text classification"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        algorithm = data.get('algorithm', 'rule_based')
        
        if not text.strip():
            return jsonify({'error': 'Please provide some text to classify'}), 400
        
        # Select classifier based on algorithm
        if algorithm == 'rule_based':
            result = rule_based_classifier.classify_text(text)
        elif algorithm == 'naive_bayes':
            result = naive_bayes_classifier.classify_text(text)
        elif algorithm == 'logistic_regression':
            result = logistic_regression_classifier.classify_text(text)
        elif algorithm == 'svm':
            result = svm_classifier.classify_text(text)
        elif algorithm == 'random_forest':
            result = random_forest_classifier.classify_text(text)
        else:
            return jsonify({'error': 'Invalid algorithm selected'}), 400
        
        if isinstance(result, str):
            return jsonify({'error': result}), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'algorithms': ['rule_based', 'naive_bayes', 'logistic_regression', 'svm', 'random_forest']
    })

if __name__ == '__main__':
    # Load or create models on startup
    load_or_create_models()
    
    print("🤖 Scikit-learn Spam Classifier Starting...")
    print("📊 Available algorithms: Rule-based, Naive Bayes, Logistic Regression, SVM, Random Forest")
    print("🚀 Server will be available at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
