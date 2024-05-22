
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load models and TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

models = {
    "MultinomialNB": pickle.load(open('model1.pkl','rb')),
    "KNeighborsClassifier": pickle.load(open('model1.pkl', 'rb')),
    "GaussianNB": pickle.load(open('model1.pkl', 'rb')),
    "ExtraTreesClassifier": pickle.load(open('model1.pkl', 'rb'))
}


ps = PorterStemmer()

# Sidebar navigation
st.sidebar.title("Spam Detection")
page = st.sidebar.radio("Select what to detect:", ("Home", "SMS Detection", "Email Detection"))

# Main content
st.title("Spam Classifier")

if page == "Home":
    st.header("Welcome to Spam Classifier App")
    st.write("""
    This app helps you detect spam messages (SMS or Email) using various machine learning models.
    To get started, select the type of message you want to detect from the sidebar.
    """)

elif page == "SMS Detection":
    st.header("SMS Spam Classifier")
    input_sms = st.text_area("Enter the SMS message")
    st.header("Model Selection")
    selected_model = st.selectbox("Select Model:", list(models.keys()))

    if st.button('Predict'):
        # Preprocess text
        transformed_input = transform_text(input_sms)
        # Vectorize text
        vector_input = tfidf.transform([transformed_input])
        vector_input_dense = vector_input.toarray()  # Convert to dense array
        # Predict using selected model
        result = models[selected_model].predict(vector_input_dense)[0]  # Pass dense array
        # Display result
        if result == 1:
            st.header("Spam")
        elif result == 0:
            st.header("Not Spam")
        else:
            st.header("")

        # Display accuracy of selected model
        accuracy = models[selected_model].score(vector_input_dense, [result])  # Pass dense array
        st.write(f"Accuracy of {selected_model}: {accuracy:.2f}")

    if st.button("Return to Home"):
        st.write("http://localhost:8502stre/?page=Home")

elif page == "Email Detection":
    st.header("Email Spam Classifier")
    input_sms = st.text_area("Enter the email message")

    # Model selection
    st.header("Model Selection")
    selected_model = st.selectbox("Select Model:", list(models.keys()))

    if st.button('Predict'):
        # Preprocess text
        transformed_input = transform_text(input_sms)
        # Vectorize text
        vector_input = tfidf.transform([transformed_input])
        vector_input_dense = vector_input.toarray()  # Convert to dense array
        # Predict using selected model
        result = models[selected_model].predict(vector_input_dense)[0]  # Pass dense array
        # Display result
        if result == 1:
            st.header("Spam")
        elif result == 0:
            st.header("Not Spam")
        else:
            st.header("")

        # Display accuracy of selected model
        accuracy = models[selected_model].score(vector_input_dense, [result])  # Pass dense array
        st.write(f"Accuracy of {selected_model}: {accuracy:.2f}")

    if st.button("Return to Home"):
        st.write("http://localhost:8502/?page=Home")


