import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score

ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    # converted to list
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


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
#model = pickle.load(open('model.pkl', 'rb'))
models = {"MultinomialNB": pickle.load(open('model1.pkl','rb')),"BernoulliNB": pickle.load(open('model1.pkl', 'rb')),
          "GaussianNB": pickle.load(open('model2.pkl', 'rb'))}

selected_model = st.selectbox("Select a model to predict:", ("MultinomialNB", "BernoulliNB", "GaussianNB"))
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    if selected_model == "MultinomialNB":
        from sklearn.naive_bayes import MultinomialNB
        model1 = MultinomialNB()



    elif selected_model == "BernoulliNB":
        # Code for Model 2 prediction
        from sklearn.naive_bayes import BernoulliNB
        model2 = BernoulliNB()

        pass
    elif selected_model == "GaussianNB":
        # Code for Model 3 prediction
        from sklearn.naive_bayes import GaussianNB
        model3 = GaussianNB()
        pass
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    elif result == 0:
        st.header("not Spam")
    else:
        st.header("")

