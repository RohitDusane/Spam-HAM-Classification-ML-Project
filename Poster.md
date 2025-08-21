Poster: Spam Classification Using Machine Learning with MLflow & Dash
1. Title

Spam Classification Using Machine Learning
With MLflow Experiment Tracking and Interactive Dash Dashboard

2. Goal

To develop and compare ML models for spam detection and visualize their performance interactively.

3. Dataset

SMS Spam Collection Dataset (UCI ML Repo)

4. Approach

Data cleaning & preprocessing of SMS messages

Training classification models

Logging metrics and parameters with MLflow

Interactive Dash dashboard for experiment visualization

5. Workflow

Data → Preprocessing → Model Training → MLflow Tracking → Dashboard Visualization

6. Key Features

Experiment selection dropdown

Metric comparison: Accuracy vs F1-score, Precision vs Recall

Summary stats of best runs

Dark theme with color-coded models

7. Results

Multiple models compared in a single dashboard

Best accuracy achieved: (add your number)

User-friendly interface for performance analysis

8. Technologies Used

Python, scikit-learn, pandas

MLflow for experiment tracking

Dash and Plotly for dashboard and visualizations

9. Applications

Spam filtering systems

Fraud detection in communication

Chatbot preprocessing

10. Learnings

Importance of reproducibility with MLflow

Building interactive dashboards for data science

Understanding model metrics and tradeoffs

11. Contact

Your Name
[LinkedIn] | [GitHub] | [Twitter]










📄 Abstract: Spam Classification using Machine Learning and NLP
1. Introduction

Spam emails and messages pose a significant threat to digital communication by wasting time, spreading misinformation, and facilitating phishing attacks. Automating the detection of such spam content is essential for secure and efficient communication systems. In this project, we build a spam classifier using Natural Language Processing (NLP) techniques and supervised machine learning models, leveraging TF-IDF vectorization and hyperparameter tuning to ensure robust and accurate predictions.

2. Materials and Methods
2.1 Dataset

A labeled dataset of SMS or email text messages categorized as "spam" or "ham" (not spam) was used for model training and evaluation. The data was pre-cleaned and balanced.

2.2 Text Preprocessing

Tokenization using nltk.word_tokenize()

Removal of stopwords (nltk.corpus.stopwords)

Stemming using Snowball Stemmer

Punctuation and number removal via regex

Word cloud visualization for spam vs. ham messages

2.3 Feature Engineering

TF-IDF (Term Frequency–Inverse Document Frequency) vectorization was applied to convert textual content into numerical vectors for model compatibility.

TF-IDF was chosen over count-based methods to emphasize important and unique terms across the corpus.

2.4 Model Selection and Training

Four different classification models were trained and compared:

Multinomial Naive Bayes (MNB)

Logistic Regression (LRC)

Random Forest Classifier (RFC)

XGBoost Classifier (XGB)

Hyperparameter tuning was conducted using GridSearchCV with cross-validation (cv=5) to identify optimal configurations for each model.

2.5 Evaluation Metrics

Performance was evaluated using:

Accuracy

Precision, Recall, and F1-Score (with emphasis on SPAM class)

ROC-AUC Curve

Confusion Matrix

2.6 Model Tracking

MLflow was used to track experiments, record hyperparameters, metrics, and visual artifacts like ROC and confusion matrices.

3. Results

XGBoost achieved the highest F1-Score and test accuracy, followed closely by Logistic Regression.

The best model (based on F1-Score for spam detection) was saved as final_model.pkl for deployment.

ROC-AUC scores exceeded 0.95, indicating strong separability between spam and ham messages.

WordClouds revealed frequent spam-related terms like “win”, “free”, “prize”, while ham messages contained more conversational language.

4. Conclusion

The project successfully demonstrates that with proper text preprocessing, TF-IDF vectorization, and model optimization, machine learning algorithms can effectively distinguish spam from non-spam messages. The deployment-ready model can now be integrated into applications to assist in real-time spam filtering.

5. Acknowledgments

We extend gratitude to the open-source community for tools like:

NLTK for natural language preprocessing

Scikit-learn for machine learning utilities

XGBoost for advanced gradient boosting

MLflow for experiment tracking and reproducibility

Special thanks to Kunal Naik
 for development insights and structured project guidance.

6. Resources

🔗 Python Web App Repository
GitHub: https://github.com/KunalNaik-Dev/Spam-HAM-Classifier

▶️ Live Web App (Flask-based)
Host locally using:






📌 Summary & Recommendation

Among all tested models, Logistic Regression (LRC) and Multinomial Naive Bayes (MNB) delivered top-tier performance, both achieving Test Accuracy of 98.55%. However, LRC edges out slightly in F1-Score (94.12%), indicating better balance between precision and recall. Notably, MNB had the highest precision (98.35%), making it slightly more conservative in spam prediction (lower false positives).

XGBoost performed well but slightly under both LRC and MNB in recall and F1, suggesting it may be overly complex for this particular dataset. Random Forest achieved perfect precision, but its recall (68.18%) was relatively low, meaning it missed many spam instances.

✅ Recommended Final Model: Logistic Regression (LRC)

Why: Strong generalization, balanced precision/recall, and highest F1-Score

Parameters: C=100, penalty='l2', max_iter=100