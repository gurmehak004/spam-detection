# spam-detection
This project implements a simple yet effective Spam/Ham (legitimate email) classification system using Machine Learning, deployed as an interactive web application with Streamlit. It allows users to evaluate the model's performance and test custom email texts for spam detection.

 Features
Model Evaluation: Displays key performance metrics (Accuracy, Classification Report, Confusion Matrix) of the trained model on a test dataset.

Interactive Prediction: Provides a text area for users to input email content and get an instant prediction (Spam or Ham).

Clear Visualizations: Uses Matplotlib and Seaborn to visualize the Confusion Matrix for better understanding of model performance.

Technologies Used
Python

Streamlit: For building the interactive web application.

Pandas: For data manipulation and loading the dataset.

Scikit-learn: For machine learning model (Logistic Regression) and TF-IDF vectorization.

Joblib: For saving and loading the trained model and TF-IDF vectorizer.

Matplotlib & Seaborn: For creating data visualizations.
