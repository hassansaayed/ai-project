# 🤖 AI Spam Classifier

A web-based spam detection system built with Python Flask and scikit-learn, featuring multiple machine learning algorithms for text classification.

## 🚀 Features

- **Multiple Algorithms**: Rule-based, Naive Bayes, Logistic Regression, SVM, and Random Forest
- **Real Dataset Training**: Uses the SMS Spam Collection Dataset
- **Web Interface**: Beautiful UI with Tailwind CSS
- **Confidence Scores**: Detailed analysis with confidence levels
- **Feature Explanations**: Understand why a message was classified as spam

## 📁 Project Structure

```
ai-spam/
├── app.py                 # Main Flask application
├── sklearn_classifiers.py # Scikit-learn classifier implementations
├── templates/
│   └── index.html        # Web interface
├── requirements.txt       # Python dependencies
├── README.md            # This file
├── ham_messages.txt     # Training data (legitimate messages)
├── spam_messages.txt    # Training data (spam messages)
└── *.pkl               # Trained model files
```

## 🛠️ Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to `http://localhost:5000`

## 🎯 Available Algorithms

### 1. **Rule-based Classifier** 🔧
- Keyword-based detection
- Pattern matching (URLs, phone numbers, etc.)
- Capitalization and punctuation analysis
- Urgency word detection

### 2. **Naive Bayes (scikit-learn)** 📊
- Multinomial Naive Bayes
- TF-IDF vectorization
- High accuracy on text classification
- Fast training and prediction

### 3. **Logistic Regression (scikit-learn)** 📈
- Linear classification with regularization
- Probability estimates
- Feature importance analysis
- Good interpretability

### 4. **SVM (scikit-learn)** ⚡
- Support Vector Machine with linear kernel
- Robust to overfitting
- Good for high-dimensional data
- Decision function scores

### 5. **Random Forest (scikit-learn)** 🌲
- Ensemble method with 100 trees
- Feature importance ranking
- Handles non-linear relationships
- Robust performance

## 📊 Performance

| Algorithm | Accuracy | Notes |
|-----------|----------|-------|
| Rule-based | ~50% | Simple but limited |
| Naive Bayes | ~98% | Excellent for text |
| Logistic Regression | ~98% | Good interpretability |
| SVM | ~99% | Robust performance |
| Random Forest | ~95% | Feature importance |

## 🎮 Usage

1. **Select an algorithm** from the dropdown menu
2. **Enter text** to classify (or try the example buttons)
3. **Click "Classify Text"** to analyze
4. **View results** with confidence scores and explanations

## 🔧 Technical Details

### **Backend (Flask)**
- RESTful API endpoints
- JSON request/response format
- Error handling and validation
- Model persistence with pickle

### **Frontend (HTML/CSS/JavaScript)**
- Responsive design with Tailwind CSS
- Real-time classification
- Interactive confidence bars
- Example text buttons

### **Machine Learning**
- **TF-IDF Vectorization**: 5,000 features, 1-2 grams
- **Cross-validation**: 80% train, 20% test split
- **Model persistence**: Automatic save/load
- **Feature engineering**: Text preprocessing

## 📈 Training Data

The system uses the **SMS Spam Collection Dataset**:
- **747 spam messages**
- **4,826 legitimate messages**
- Real-world SMS data
- Proper preprocessing and cleaning

## 🚀 API Endpoints

- `GET /` - Main web interface
- `POST /classify` - Text classification API
- `GET /health` - Health check endpoint

### Example API Request:
```json
{
  "text": "URGENT! You have won a free iPhone!",
  "algorithm": "naive_bayes"
}
```

### Example API Response:
```json
{
  "prediction": "spam",
  "confidence": 95.2,
  "spam_score": 9.5,
  "reasons": ["Key indicator: urgent", "Key indicator: free"],
  "algorithm": "naive_bayes"
}
```

## 🎉 Key Benefits

- ✅ **Multiple algorithms** for comparison
- ✅ **Real dataset training** (not just examples)
- ✅ **Professional scikit-learn implementation**
- ✅ **Beautiful web interface**
- ✅ **Detailed explanations** for classifications
- ✅ **Confidence scores** and spam ratings
- ✅ **Easy to use** and understand

## 🔮 Future Enhancements

- [ ] Add more algorithms (XGBoost, Neural Networks)
- [ ] Implement model ensemble voting
- [ ] Add real-time learning from user feedback
- [ ] Create API documentation
- [ ] Add model performance metrics dashboard
- [ ] Support for different languages

---

**Built with ❤️ using Python, Flask, and scikit-learn**

