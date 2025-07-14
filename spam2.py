import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # For saving and loading the trained model and vectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# --- Load the Model and Vectorizer ---
try:
    model = joblib.load('spam_detection_model.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    st.error("Error: Trained model and vectorizer files not found. Please train the model first.")
    st.stop()

# --- Load the Dataset (for evaluation) ---
try:
    df = pd.read_csv("C:\\Users\\gurme\\Downloads\\mail_data.csv")  
except FileNotFoundError:
    st.error("Error: 'mail_data.csv' not found. Make sure the file is in the correct directory.")
    st.stop()

# Preprocess the dataset
mail_data = df.where((pd.notnull(df)), '')
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
mail_data['Category'] = mail_data['Category'].astype(int)  
Y = mail_data['Category']
X = mail_data['Message']

# Split data for evaluation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3, stratify=Y)

# --- Feature Extraction (using the loaded vectorizer) ---
X_test_features = tfidf_vectorizer.transform(X_test)

# --- Streamlit App ---
st.title("Spam/Ham Email Detector")

# --- 1.  Evaluation Metrics ---
st.header("Model Evaluation")
st.write("Evaluating the model's performance on a held-out test set:")

# Predictions
Y_pred = model.predict(X_test_features)

# Calculate and display metrics
accuracy = accuracy_score(Y_test, Y_pred)
st.write(f"**Accuracy:** {accuracy:.4f}")

st.subheader("Classification Report:")
report = classification_report(Y_test, Y_pred, target_names=['Spam', 'Ham'])  # Add target names for clarity
# Add extra newlines for better spacing
report_lines = report.split('\n')
spaced_report = '\n'.join([line.rstrip() + '  ' for line in report_lines])
st.text(spaced_report)

# Confusion Matrix
st.subheader("Confusion Matrix:")
conf_matrix = confusion_matrix(Y_test, Y_pred)
fig, ax = plt.subplots()  # Create figure and axes for Matplotlib
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Spam', 'Ham'], yticklabels=['Spam', 'Ham'], ax=ax)  # Pass the axes to heatmap
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
st.pyplot(fig)  # Display the Matplotlib figure in Streamlit

# --- 2.  Interactive Prediction ---
st.header("Email Classification")
email_text = st.text_area("Enter email text here:", height=200)

if st.button("Detect Spam/Ham"):
    if email_text:
        # Preprocess the input email using the *loaded* vectorizer
        input_mail_features = tfidf_vectorizer.transform([email_text])

        # Make the prediction
        prediction = model.predict(input_mail_features)[0]

        # Display the result
        st.subheader("Prediction:")
        if prediction == 1:
            st.success("This email is likely HAM (Not Spam)")
        else:
            st.error("This email is likely SPAM")
    else:
        st.warning("Please enter some email text.")
