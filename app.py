import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
import string

# Load the data
file_path = 'Raw_Reviews.csv'
reviews_df = pd.read_csv(file_path)

# Handle missing values in the Text_Review column by dropping these rows
reviews_df.dropna(subset=['Text_Review'], inplace=True)

# Fill missing titles with an empty string
reviews_df['Title'].fillna('', inplace=True)

# Normalize text data: convert to lowercase and remove punctuation
def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

reviews_df['Text_Review'] = reviews_df['Text_Review'].apply(normalize_text)
reviews_df['Title'] = reviews_df['Title'].apply(normalize_text)

# Combine Title and Text_Review for a comprehensive text feature
reviews_df['Full_Review'] = reviews_df['Title'] + ' ' + reviews_df['Text_Review']

# Separate rows with known and unknown 'Type'
known_type_df = reviews_df.dropna(subset=['Type'])
unknown_type_df = reviews_df[reviews_df['Type'].isna()]

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(known_type_df['Full_Review'])
y = known_type_df['Type']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)

# Impute the missing 'Type' values
X_unknown = vectorizer.transform(unknown_type_df['Full_Review'])
unknown_type_df.loc[:, 'Type'] = clf.predict(X_unknown)

# Combine the data back together
imputed_reviews_df = pd.concat([known_type_df, unknown_type_df])

# Remove Full_Review column
cleaned_reviews_df = imputed_reviews_df.drop('Full_Review', axis=1)

# Sentiment analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

cleaned_reviews_df['Sentiment'] = cleaned_reviews_df['Text_Review'].apply(get_sentiment)
cleaned_reviews_df['Sentiment_Category'] = pd.cut(cleaned_reviews_df['Sentiment'], bins=[-1, -0.1, 0.1, 1], labels=['Negative', 'Neutral', 'Positive'])

# Sidebar
st.sidebar.title("Review Analysis")
option = st.sidebar.selectbox("Choose Analysis", ["Overview", "Word Clouds", "Detailed Review"])

# Overview
if option == "Overview":
    st.title("Overview of Reviews")
    
    st.subheader("Distribution of Ratings")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Rating', data=cleaned_reviews_df, palette='viridis', hue=None, legend=False)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    st.pyplot(plt)
    
    st.subheader("Distribution of Sentiment Categories")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Sentiment_Category', data=cleaned_reviews_df, palette='viridis', hue=None, legend=False)
    plt.title('Distribution of Sentiment Categories')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Count')
    st.pyplot(plt)

# Word Clouds
elif option == "Word Clouds":
    st.title("Word Clouds of Reviews")
    
    positive_reviews = ' '.join(cleaned_reviews_df[cleaned_reviews_df['Sentiment_Category'] == 'Positive']['Text_Review'])
    negative_reviews = ' '.join(cleaned_reviews_df[cleaned_reviews_df['Sentiment_Category'] == 'Negative']['Text_Review'])
    
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
    wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)
    
    st.subheader("Word Cloud for Positive Reviews")
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud_positive, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    
    st.subheader("Word Cloud for Negative Reviews")
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Detailed Review
elif option == "Detailed Review":
    st.title("Detailed Review Analysis")
    
    custom_stopwords = set([
        'the', 'and', 'i', 'it', 'a', 'is', 'to', 'this', 'in', 'was', 'of', 'but', 'not', 'on', 'for', 'so', 'my', 'like', 'with', 'that', 'they', 'you', 'as', 'at', 'be'
    ])
    
    def get_common_words(text):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in custom_stopwords]
        word_counts = Counter(filtered_words)
        return word_counts.most_common(20)
    
    st.subheader("Common Words in Negative Reviews")
    negative_words = ' '.join(cleaned_reviews_df[cleaned_reviews_df['Sentiment_Category'] == 'Negative']['Text_Review'])
    common_negative_words = get_common_words(negative_words)
    common_negative_df = pd.DataFrame(common_negative_words, columns=["Word", "Frequency"])
    st.table(common_negative_df)
    
    st.subheader("Common Words in Positive Reviews")
    positive_words = ' '.join(cleaned_reviews_df[cleaned_reviews_df['Sentiment_Category'] == 'Positive']['Text_Review'])
    common_positive_words = get_common_words(positive_words)
    common_positive_df = pd.DataFrame(common_positive_words, columns=["Word", "Frequency"])
    st.table(common_positive_df)
