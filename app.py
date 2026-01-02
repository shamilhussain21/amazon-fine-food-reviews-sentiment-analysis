import streamlit as st
import joblib
import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

# ------------------------
# NLTK downloads (for Streamlit Cloud)
# ------------------------
nltk.download('stopwords')
nltk.download('wordnet')

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="Amazon Reviews Sentiment Analysis",
    page_icon="🍴",
    layout="wide"
)

# ------------------------
# Load model and vectorizer (from models folder)
# ------------------------
model = joblib.load("models/svm_amazon_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stopwords_set = set(stopwords.words('english'))

# ------------------------
# Text preprocessing function
# ------------------------
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Keep only letters
    text = re.sub("[^A-Za-z]+", " ", text)
    # Lemmatize & remove stopwords
    text = " ".join(
        lemmatizer.lemmatize(word.lower())
        for word in text.split()
        if word.lower() not in stopwords_set
    )
    return text

# ------------------------
# App header
# ------------------------
st.title("🍴 Amazon Fine Food Reviews Sentiment Analysis")
st.write("Enter a review or upload a CSV file to predict sentiment.")
st.markdown("---")

# ------------------------
# Single review prediction
# ------------------------
st.subheader("Review Prediction")
user_input = st.text_area("Type your review here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to predict.")
    else:
        processed_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([processed_input])
        prediction = model.predict(input_vector)[0]

        # Sentiment
        sentiment = "Positive" if prediction == 1 else "Negative"
        color = "green" if prediction == 1 else "red"

        # Confidence score
        if hasattr(model, "decision_function"):
            score = model.decision_function(input_vector)[0]
            confidence = 1 / (1 + 2.71828**(-score))
        else:
            confidence = model.predict_proba(input_vector)[0][prediction]

        st.markdown(
            f"<h3 style='color:{color}'>Predicted Sentiment: {sentiment}</h3>",
            unsafe_allow_html=True
        )
        st.info(f"Confidence Score: {confidence:.2f}")

st.markdown("---")

# ------------------------
# Batch prediction from CSV
# ------------------------
st.subheader("Batch Prediction from CSV")
st.write("Upload a CSV file and select the column containing reviews.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Let user select the review column dynamically
        review_columns = df.columns.tolist()
        selected_column = st.selectbox("Select the column containing reviews:", review_columns)

        if selected_column:
            st.info(f"Processing {len(df)} reviews from column '{selected_column}'...")

            # Preprocess and predict
            df['processed_text'] = df[selected_column].apply(preprocess_text)
            X = vectorizer.transform(df['processed_text'])
            df['Sentiment'] = model.predict(X)
            df['Sentiment'] = df['Sentiment'].apply(lambda x: "Positive" if x == 1 else "Negative")

            # Display results
            st.success("Predictions completed!")
            st.dataframe(df[[selected_column, 'Sentiment']])

            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")






























