import streamlit as st
import re
from nltk.corpus import stopwords
import joblib
nltk.download('stopwords')

custom_stopwords = [
    'according', 'administration', 'also', 'america', 'american', 'americans', 'another',
    'back', 'bill', 'black', 'called', 'campaign', 'clinton', 'could',
    'country', 'day', 'department', 'donald', 'election', 'even', 'every', 'fact', 'first',
    'former', 'fox', 'get', 'go', 'going', 'good', 'government', 'group', 'hillary',
    'house', 'image', 'it', 'know', 'last', 'law', 'like', 'made', 'make', 'man', 'many',
    'may', 'media', 'much', 'national', 'never', 'new', 'news', 'obama', 'office',
    'one', 'party', 'people', 'police', 'political', 'president', 'presidential', 'public',
    'really', 'republican', 'republicans', 'right', 'said', 'say', 'says', 'see',
    'show', 'since', 'state', 'states', 'still', 'support', 'take', 'think', 'time',
    'told', 'trump', 'two', 'united', 'us', 'via', 'video', 'vote', 'want', 'way', 'well',
    'white', 'women', 'world', 'would', 'year', 'years'
]

# Load the trained model
model = joblib.load('nlpmodel.pkl')

# Define a function to preprocess the text
def preprocess_text(text):
    text = re.sub(r'\W+|\d+', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove stopwords
    stop_words = set(stopwords.words('english') + custom_stopwords)
    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text

# Define the Streamlit app
def main():
    # Add a title and a brief description
    st.title("Text Classification Demo")
    st.write("Enter some text and click the 'Predict' button to classify it as Fake or True.")

    # Create a text input box for user input
    text_input = st.text_area("Enter text", "")

    # Add a predict button
    if st.button("Predict"):
        # Preprocess the input text
        preprocessed_text = preprocess_text(text_input)

        # Make a prediction using the loaded model
        prediction = model.predict([preprocessed_text])[0]

        # Map prediction to 'Fake' or 'True'
        prediction_label = "Fake" if prediction == 0 else "True"

        # Display the prediction
        st.subheader("Prediction")
        st.write(prediction_label)

# Run the app
if __name__ == '__main__':
    main()
