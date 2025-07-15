import streamlit as st
import joblib

# Load the pre-trained model and vectorizer
model = joblib.load('text_classification.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# App layout
st.title('Review Sentiment Classifier')
st.write('Enter a movie review to classify it as positive or negative.')

# Text area for user input
review = st.text_area('Review:', '')

# Prediction button
if st.button('Predict'):
    if review:
        # Transform the input text using the vectorizer
        review_vector = vectorizer.transform([review])
        
        # Make prediction using the model
        prediction = model.predict(review_vector)
        
        # Determine sentiment
        sentiment = 'Positive' if prediction[0] == 'positive' else 'Negative'
        
        # Display the result
        st.write(f'The review is: {sentiment}')
    else:
        st.write('Please enter a review.')