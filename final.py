import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

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
with open('vectorizer.pkl', 'rb') as f:tfidf = pickle.load(f)

#with open('model.pkl', 'rb') as f:models = pickle.load(f)
models = {"MultinomialNB": pickle.load(open('model1.pkl','rb'))," KNeighborsClassifier": pickle.load(open('model1.pkl', 'rb')),
          "GaussianNB": pickle.load(open('model1.pkl', 'rb')),"  ExtraTreesClassifier": pickle.load(open('model1.pkl', 'rb'))}
# Sidebar navigation
st.sidebar.title("Spam Detection")
page = st.sidebar.selectbox("Select what to detect:", ("SMS Detection", "Email Detection"))

# Main content
st.title("Spam Classifier")

ps = PorterStemmer()

if page == "SMS Detection":
    st.header("SMS Spam Classifier")
    input_sms = st.text_area("Enter the SMS message")

elif page == "Email Detection":
    st.header("Email Spam Classifier")
    input_sms = st.text_area("Enter the email message")
st.sidebar.title("select model")
selected_model = st.sidebar.selectbox(
    "Select Model:",
    list(models.keys()))

if st.button('Predict'):
    # Preprocess text
    transformed_input = transform_text(input_sms)
    # Vectorize text
    vector_input = tfidf.transform([transformed_input])
    vector_input_dense = vector_input.toarray()  # Convert to dense array
    print("Shape of vector_input_dense:", vector_input_dense.shape)  # Print the shape
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
    st.write(f"Accuracy of {selected_model}: {accuracy}")