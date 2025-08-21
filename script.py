# Imports
import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score,roc_curve,auc, confusion_matrix
from xgboost import XGBClassifier
import pickle
import mlflow
from wordcloud import WordCloud

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv(r'Spam_HAM\data\smsspamcollection\SMSSpamCollection', sep='\t', names=['target', 'text'])
print(df.head())

# Data Exploration
print(f"Data shape: {df.shape}")
print('-' * 35)
print(f"Data info : {df.info()}")
print('-' * 35)
print(f"Missing in data:\n{df.isnull().sum()}")
print('-' * 35)
print(f"Duplicates in data: {df.duplicated().sum()}")
print('-' * 35)

# Label Encoding
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Feature Extraction
# Add new columns for character count, word count, and sentence count
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sent'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# Descriptive Stats
print(df[['num_characters', 'num_words', 'num_sent']].describe().T)
print(df[df['target'] == 0][['num_characters', 'num_words', 'num_sent']].describe().T)  # HAM stats
print(df[df['target'] == 1][['num_characters', 'num_words', 'num_sent']].describe().T)  # SPAM stats

# Visualization of Target Distribution
plt.pie(df.target.value_counts(), labels=['ham', 'spam'], autopct="%.2f")
plt.title('Target Distribution')
plt.show()

# Visualizing Feature Distributions
COLS = df[['num_characters', 'num_words', 'num_sent']]
plt.figure(figsize=(18, 5))
for i, col in enumerate(COLS, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[df['target'] == 0][col], color='skyblue', label='HAM')
    sns.histplot(df[df['target'] == 1][col], color='darkorange', label='SPAM')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.legend()
plt.tight_layout()
plt.show()

# Preprocessing Function
snow = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

def preprocess_text(text_series):
    corpus = []
    for text in text_series:
        review = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-letter characters
        review = review.lower()  # Lowercase
        review = word_tokenize(review)  # Tokenization
        review = [snow.stem(word) for word in review if word not in stop_words]  # Stemming
        corpus.append(" ".join(review))  # Join tokens back to string
    return corpus

df['cleaned_text'] = preprocess_text(df['text'])

# WordCloud for HAM and SPAM messages
wc = WordCloud(width=800, height=500, min_font_size=10, background_color='black')
plt.figure(figsize=(18, 8))

# HAM Wordcloud
plt.subplot(1, 2, 1)
ham_words = df[df['target'] == 0]['cleaned_text'].str.cat(sep=" ")
ham_wc = wc.generate(ham_words)
plt.imshow(ham_wc, interpolation='bilinear')
plt.axis('off')
plt.title("HAM Messages WordCloud")

# SPAM Wordcloud
plt.subplot(1, 2, 2)
spam_words = df[df['target'] == 1]['cleaned_text'].str.cat(sep=" ")
spam_wc = wc.generate(spam_words)
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis('off')
plt.title("SPAM Messages WordCloud")

plt.tight_layout()
plt.show()

# Vectorization: BoW and TF-IDF
cv_binary = CountVectorizer(max_features=3000, ngram_range=(1, 2), binary=True)
X_bow = cv_binary.fit_transform(df['cleaned_text']).toarray()

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df['cleaned_text']).toarray()

print("BoW matrix shape:", X_bow.shape)
print("TF-IDF matrix shape:", X_tfidf.shape)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['target'], test_size=0.3, random_state=24)
print(f"Train set shape: {X_train.shape} , {y_train.shape}")
print(f"Test set  shape: {X_test.shape} , {y_test.shape}")

# Define the models
models = {
    'mnb' : MultinomialNB(),
    'rfc' : RandomForestClassifier(n_estimators=50, random_state=2),
    'etc' : ExtraTreesClassifier(n_estimators=50, random_state=2),
    'xgb' : XGBClassifier(n_estimators=50, random_state=2),
}

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri(r'file:///E:/_DataScienc_KNaik/NLP/Spam_HAM/mlruns')
mlflow.set_experiment('Exp_1')

# Initialize a list to store model reports
report = []

# Loop through each model, train, evaluate, and log metrics
for name, model in models.items():
    with mlflow.start_run() as run:
        print(f"MLflow Run started with run_id: {run.info.run_id}")
        print(f"Training {name} model...")
        
        # Train the model
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Save the model as a pickle file
        with open(f"models/{name}_Exp_1_model.pkl", 'wb') as f:
            pickle.dump(model, f)

        # Evaluate the model
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        # Get classification report
        class_report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=1)
        precision = class_report['1']['precision']
        recall = class_report['1']['recall']
        f1_score = class_report['1']['f1-score']

        report.append({
            'Model': name,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Precision (SPAM)': precision,
            'Recall (SPAM)': recall,
            'F1-Score (SPAM)': f1_score
        })

        # Log metrics to MLflow
        mlflow.log_param("Model Name", name)
        mlflow.log_metric("Train Accuracy", train_acc)
        mlflow.log_metric("Test Accuracy", test_acc)
        mlflow.log_metric("Precision_SPAM", precision)
        mlflow.log_metric("Recall_SPAM", recall)
        mlflow.log_metric("F1-Score_SPAM", f1_score)

        # Log the model itself
        mlflow.sklearn.log_model(model, name=f"{name}_model")

        # Generate and log the confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_image_path = f'artifacts/images/cm_Exp_1_{name}.png'
        plt.savefig(cm_image_path)
        plt.close()
        mlflow.log_artifact(cm_image_path)

        # Generate and log the ROC curve
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {name}')
        plt.legend(loc='lower right')
        roc_image_path = f'artifacts/roc/roc_Exp_1_{name}.png'
        plt.savefig(roc_image_path)
        plt.close()
        mlflow.log_artifact(roc_image_path)

# After all models are trained and logged, create the final performance report
reports_df = pd.DataFrame(report).sort_values(by='F1-Score (SPAM)', ascending=False)
reports_df.to_csv('report/Exp_1_performance.csv', index=False)

# Print the final performance report
print(reports_df)

# Model Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Define the models and parameter grids
models = {
    'mnb': MultinomialNB(),
    'rfc': RandomForestClassifier(n_estimators=50, random_state=2),
    'xgb': XGBClassifier(n_estimators=50, random_state=2),
}

param_grids = {
    'mnb': {
        'alpha': [0.1, 0.5, 1.0]
    },
    'rfc': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'xgb': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 5, 7]
    }
}

# Set the MLflow tracking URI and experiment
mlflow.set_tracking_uri(r'file:///E:/_DataScienc_KNaik/NLP/Spam_HAM/mlruns')
mlflow.set_experiment('Exp_2_HT')

best_model = None
best_model_name = ""
best_accuracy = 0
report = []

# Iterate through models and perform hyperparameter tuning
for name, model in models.items():
    with mlflow.start_run() as run:  # Start a new MLflow run for each model
        print(f"MLflow Run started with run_id: {run.info.run_id}")
        print(f"Training model: {name}")
        
        # Apply GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_model_found = grid_search.best_estimator_

        print(f"Best parameters for {name}: {best_params}")

        # Save the tuned model
        with open(f"models/tuned_{name}.pkl", 'wb') as f:
            pickle.dump(best_model_found, f)

        # Predict using the tuned model
        y_pred_train = best_model_found.predict(X_train)
        y_pred_test = best_model_found.predict(X_test)

        # Evaluate the model
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        # Generate classification report
        class_report = classification_report(y_test, y_pred_test, output_dict=True)
        precision = class_report['1']['precision']
        recall = class_report['1']['recall']
        f1_score = class_report['1']['f1-score']

        # Log metrics and the tuned model to MLflow
        mlflow.log_param('model_name', name)
        mlflow.log_param('best_params', best_params)
        mlflow.log_metric('Train Accuracy', train_acc)
        mlflow.log_metric('Test Accuracy', test_acc)
        mlflow.log_metric('Precision SPAM', precision)
        mlflow.log_metric('Recall SPAM', recall)
        mlflow.log_metric('F1-Score SPAM', f1_score)

        # Log the tuned model
        mlflow.sklearn.log_model(best_model_found, name=f'model_{name}')

        # Generate and save confusion matrix image
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_image_path = f'artifacts/images/cm_Exp_2_{name}.png'
        plt.savefig(cm_image_path)
        plt.close()
        
        # Log confusion matrix image to MLflow
        mlflow.log_artifact(cm_image_path)

        # Generate and save ROC curve image
        fpr, tpr, _ = roc_curve(y_test, best_model_found.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {name}')
        plt.legend(loc='lower right')
        roc_image_path = f'artifacts/roc/roc_Exp_2_{name}.png'
        plt.savefig(roc_image_path)
        plt.close()

        # Log ROC curve image to MLflow
        mlflow.log_artifact(roc_image_path)

        # Compare models and save the best one
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_name = name
            best_model = best_model_found

        # Append results to the report list
        report.append({
            'Model': name,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Precision (SPAM)': precision,
            'Recall (SPAM)': recall,
            'F1-Score (SPAM)': f1_score,
            'Best Params': str(best_params)
        })

# Final Report: Save the results in a DataFrame
reports_df = pd.DataFrame(report)
reports_df.to_csv('report/tuned_model_performance.csv', index=False)

# Save the best model as the final model
with open(f"models/final_model.pkl", 'wb') as f:
    pickle.dump(best_model, f)

print(f"Best model saved as final_model.pkl: {best_model_name} with accuracy: {best_accuracy}")