# Spam Classification Model with Multiple Machine Learning Algorithms

## GOAL
The goal of this project is to develop a **spam classification model** that can classify SMS messages as either **spam** or **ham** (not spam). The project uses multiple machine learning models like **Logistic Regression**, **Random Forest**, **Decision Tree**, **XGBoost**, **Multinomial Naive Bayes**, and **Voting Classifier** to obtain the final model, which is then deployed using a **Flask app** on **Render** for real-time predictions.

## PURPOSE
The purpose of this project is to explore different machine learning models and their performance in spam classification tasks, then deploy the final model in a Flask web application so that it can classify messages dynamically via API requests.

## DATASET
The dataset used in this project is the **SMS Spam Collection Dataset**, available at:  
[SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

This dataset contains labeled SMS messages with two classes: **spam** and **ham**.

## DESCRIPTION
This project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to convert SMS text messages into numerical features. We then apply multiple machine learning models (Logistic Regression, Random Forest, Decision Tree, XGBoost, Multinomial Naive Bayes) and combine their results using a **Voting Classifier** to create a final predictive model. The model is then deployed in a Flask app, making it available for real-time predictions.

## WHAT I HAD DONE
- Acquired and loaded the dataset.
- Cleaned and preprocessed the text data (lowercasing, removing stop words, stemming).
- Applied **TF-IDF** vectorization to transform text into numerical features.
- Implemented multiple machine learning algorithms (Logistic Regression, Random Forest, Decision Tree, XGBoost, Multinomial Naive Bayes).
- Combined the models using **Voting Classifier** to create the final predictive model.
- Developed a **Flask API** to serve the model and deployed it on **Render**.

## WORKFLOW OF YOUR PROJECT FILES
1. **Data Acquisition**: Loaded the SMS dataset and displayed the first few rows.
2. **Data Preprocessing**: Cleaned and tokenized the text data.
3. **TF-IDF Vectorization**: Converted the cleaned text data into numerical features using TF-IDF.
4. **Model Training**: Trained multiple models (Logistic Regression, Random Forest, Decision Tree, XGBoost, Naive Bayes).
5. **Voting Classifier**: Combined all the models into a final **Voting Classifier**.
6. **Model Evaluation**: Evaluated the models using **accuracy**, **precision**, **recall**, and **F1-score**.
7. **Flask App**: Developed a Flask web application to make predictions.
8. **Deployment**: Deployed the Flask app on **Render** for real-time usage.

## DIAGRAMS
- **Data Preprocessing Flow**:  
  ![Data Preprocessing Flowchart](images/data_preprocessing_flow.png)

- **Model Comparison**:  
  ![Model Comparison](images/model_comparison.png)

## STATE YOUR PROCEDURE AND UNDERSTANDING FROM YOUR WORK
1. **Preprocessing**: The text data was cleaned by removing unwanted characters, converting all text to lowercase, and stemming using **SnowballStemmer**. We also removed **stopwords** to enhance feature quality.
2. **TF-IDF Vectorization**: I chose **TF-IDF** for feature extraction because it captures the importance of words within documents in relation to the entire corpus, making it a strong technique for text classification tasks.
3. **Model Selection**: I experimented with multiple classifiers and found that combining models using a **Voting Classifier** increased prediction accuracy.
4. **Deployment**: I used **Flask** to wrap the final model into an API and deployed it on **Render** for real-time access.
   
### What I learned:
- Different models can have varying levels of performance on the same dataset.
- Combining multiple models through a **Voting Classifier** improves the overall prediction power.

## DETAILED EXPLANATION OF SCRIPT, IF APPLICABLE
1. **Data Preprocessing**:
   - Text is tokenized, stopwords are removed, and stemming is applied using **NLTK**.
   
2. **TF-IDF Vectorization**:
   - **TfidfVectorizer** from **Scikit-learn** is used to convert text into numerical features.

3. **Model Training**:
   - Trained the following models:
     - **Logistic Regression**
     - **Random Forest**
     - **Decision Tree**
     - **XGBoost**
     - **Multinomial Naive Bayes**
   
4. **Model Evaluation**:
   - Evaluated models using **classification_report** for accuracy, precision, recall, and F1-score.
   
5. **Voting Classifier**:
   - Combined all models using **VotingClassifier** for improved performance.

6. **Flask App**:
   - Created a REST API using **Flask** to serve the model and make predictions.

## USAGE
The issue of **spam classification** is important in filtering out unwanted messages in many domains like **SMS**, **emails**, and **social media messages**. This solution provides a reliable, real-time method for detecting spam messages based on a variety of machine learning models.

## USE CASES
1. **SMS Spam Detection**: Automatically filter SMS messages that are spam.
2. **Email Spam Filtering**: Classify emails as spam or not.
3. **Real-time Applications**: Use in customer service to filter out spammy inquiries.

## LIBRARIES USED
- **Pandas**: For data manipulation.
- **NLTK**: For natural language processing (tokenization, stopwords, stemming).
- **Scikit-learn**: For model building, evaluation, and vectorization.
- **XGBoost**: For boosting-based classification.
- **Flask**: For creating the API and serving the model.
- **Matplotlib/Seaborn**: For data visualization.

## ADVANTAGES
- **Real-time predictions** for spam classification via the Flask API.
- **Multiple model comparison** to choose the best-performing algorithm.
- **Scalable**: Can be extended to classify emails or other types of messages.

## DISADVANTAGES
- **Imbalanced dataset**: If the dataset is highly imbalanced, some models may perform poorly.
- **Training time**: Training models like **XGBoost** and **Random Forest** can be time-consuming.

## APPLICATIONS
1. **SMS Filtering**: For mobile operators or messaging services to filter spam messages.
2. **Email Filtering**: To block spam emails in email clients like Gmail, Yahoo, etc.
3. **Customer Service**: Spam detection in automated customer support systems.
4. **Social Media**: Detect spam or malicious messages on platforms like Twitter, Facebook, etc.

## RESEARCH
I have researched various spam detection techniques and found that using **TF-IDF** with models like **Logistic Regression** and **XGBoost** yields great results for spam classification tasks. 

I explored using multiple models and evaluated their performance to choose the best approach for this problem.

## SCREENSHOTS
1. **Prediction Interface**:
   ![Prediction Interface](images/prediction_interface.png)

2. **Model Evaluation**:
   ![Model Evaluation](images/model_evaluation.png)

## CONCLUSION
This project successfully builds and deploys a **spam classification model** using multiple machine learning algorithms, evaluated through performance metrics, and deployed via **Flask** on **Render**. The **Voting Classifier** ensures a higher accuracy by combining the strengths of multiple models. The real-time API implementation allows for efficient and easy deployment for production environments.

## REFERENCES
1. **Scikit-learn Documentation**: https://scikit-learn.org/stable/
2. **NLTK Documentation**: https://www.nltk.org/
3. **Kaggle Dataset**: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
4. **Flask Documentation**: https://flask.palletsprojects.com/

## YOUR NAME
ROHIT R DUSANE   
GitHub: ['Github'](https://github.com/RohitDusane)  


## DISCLAIMER, IF ANY
This project is for educational purposes only. The model is based on the **SMS Spam Collection Dataset**, which may not reflect real-world spam classification challenges in production systems.
