import streamlit as st
from textblob import TextBlob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cleantext
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Reviews.csv')

analyzer = SentimentIntensityAnalyzer()

sentiment_scores = []
review_text = df['Text']
blob_subj = []
for review in review_text:
    sentiment_scores.append(analyzer.polarity_scores(review)["compound"])
    blob = TextBlob(review)
    blob_subj.append(blob.subjectivity)

sentiment_classes = []
for sentiment_score in sentiment_scores:
    if sentiment_score > 0.8:
        sentiment_classes.append("Highly Positive")
    elif sentiment_score > 0.4:
        sentiment_classes.append("Positive")
    elif -0.4 <= sentiment_score <= 0.4:
        sentiment_classes.append("Neutral")
    elif sentiment_score < -0.4:
        sentiment_classes.append("Negative")
    else:
        sentiment_classes.append("Highly Negative")

st.title("Sentiment Analysis on Customer Feedback")

user_input = st.text_area("Enter the Feedback: ")
blob = TextBlob(user_input)

user_sentiment_score = analyzer.polarity_scores(user_input)['compound']

if user_sentiment_score > 0.8:
    user_sentiment_class = "Highly Positive"
elif user_sentiment_score > 0.4:
    user_sentiment_class = "Positive"
elif -0.4 <= user_sentiment_score <= 0.4:
    user_sentiment_class = "Neutral"
elif user_sentiment_score < -0.4:
    user_sentiment_class = "Negative"
else:
    user_sentiment_class = "Highly Negative"

st.write("**VADER Sentiment Class:**", user_sentiment_class)
st.write("**VADER Sentiment Score:**", user_sentiment_score)
st.write("**TextBlob Polarity:**", blob.sentiment.polarity)
st.write("**TextBlob Subjectivity:**", blob.sentiment.subjectivity)

pre = st.text_input('Clean Text:')
if pre:
    st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True))
else:
    st.write("No text is being provided by the user for cleaning.")

st.subheader("Graphical Representation of Data")
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x=sentiment_scores, hue=sentiment_classes, multiple="stack", bins=30)
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.title("Score Distribution by Class")
st.pyplot(plt)

df["Sentiment Class"] = sentiment_classes
df["Sentiment Score"] = sentiment_scores
df["Subjectivity"] = blob_subj

new_df = df[["Score", "Text", "Sentiment Score", "Sentiment Class", "Subjectivity"]]
st.subheader("Input DataFrame")
st.dataframe(new_df.head(10), use_container_width=True)

st.sidebar.header("About")
st.sidebar.info("This app analyzes customer feedback using VADER and TextBlob to classify sentiment and visualize the data.")
st.sidebar.header("Settings")
theme = st.sidebar.selectbox("Choose Theme", ["Default", "Dark", "Light"])

if theme == "Dark":
    st.markdown(
        """
        <style>
        .main {background-color: #333333; color: white;}
        .sidebar .sidebar-content {background-color: #444444;}
        </style>
        """,
        unsafe_allow_html=True
    )
elif theme == "Light":
    st.markdown(
        """
        <style>
        .main {background-color: #f4f4f4; color: black;}
        .sidebar .sidebar-content {background-color: #ffffff;}
        h1, h2, h3, h4, h5, h6 {color: black;}
        </style>
        """,
        unsafe_allow_html=True
    )


