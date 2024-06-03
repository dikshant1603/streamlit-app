{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "936c8982",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\HP\\AppData\\Local\\Temp\\ipykernel_2008\\3283962846.py:58: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  unknown_type_df['Type'] = clf.predict(X_unknown)\n",
      "C:\\Users\\HP\\AppData\\Local\\Temp\\ipykernel_2008\\3283962846.py:83: FutureWarning: \n",
      "\n",
      "Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.\n",
      "\n",
      "  sns.countplot(x='Rating', data=cleaned_reviews_df, palette='viridis')\n",
      "C:\\Users\\HP\\AppData\\Local\\Temp\\ipykernel_2008\\3283962846.py:91: FutureWarning: \n",
      "\n",
      "Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.\n",
      "\n",
      "  sns.countplot(x='Sentiment_Category', data=cleaned_reviews_df, palette='viridis')\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "from wordcloud import WordCloud\n",
    "from textblob import TextBlob\n",
    "from collections import Counter\n",
    "\n",
    "# Load the data\n",
    "file_path = 'Downloads/Raw_Reviews.csv'\n",
    "reviews_df = pd.read_csv(file_path)\n",
    "\n",
    "# Handle missing values in the Text_Review column by dropping these rows\n",
    "reviews_df.dropna(subset=['Text_Review'], inplace=True)\n",
    "\n",
    "# Fill missing titles with an empty string\n",
    "reviews_df['Title'].fillna('', inplace=True)\n",
    "\n",
    "# Normalize text data: convert to lowercase and remove punctuation\n",
    "import string\n",
    "\n",
    "def normalize_text(text):\n",
    "    text = text.lower()\n",
    "    text = text.translate(str.maketrans('', '', string.punctuation))\n",
    "    return text\n",
    "\n",
    "reviews_df['Text_Review'] = reviews_df['Text_Review'].apply(normalize_text)\n",
    "reviews_df['Title'] = reviews_df['Title'].apply(normalize_text)\n",
    "\n",
    "# Combine Title and Text_Review for a comprehensive text feature\n",
    "reviews_df['Full_Review'] = reviews_df['Title'] + ' ' + reviews_df['Text_Review']\n",
    "\n",
    "# Separate rows with known and unknown 'Type'\n",
    "known_type_df = reviews_df.dropna(subset=['Type'])\n",
    "unknown_type_df = reviews_df[reviews_df['Type'].isna()]\n",
    "\n",
    "# Vectorize the text data\n",
    "vectorizer = TfidfVectorizer(max_features=1000)\n",
    "X = vectorizer.fit_transform(known_type_df['Full_Review'])\n",
    "y = known_type_df['Type']\n",
    "\n",
    "# Split the data into training and test sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Train a RandomForest classifier\n",
    "clf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "# Evaluate the model\n",
    "y_pred = clf.predict(X_test)\n",
    "\n",
    "# Impute the missing 'Type' values\n",
    "X_unknown = vectorizer.transform(unknown_type_df['Full_Review'])\n",
    "unknown_type_df['Type'] = clf.predict(X_unknown)\n",
    "\n",
    "# Combine the data back together\n",
    "imputed_reviews_df = pd.concat([known_type_df, unknown_type_df])\n",
    "\n",
    "#Remove full reviews column\n",
    "cleaned_reviews_df=imputed_reviews_df.drop('Full_Review', axis=1)\n",
    "\n",
    "# Sentiment analysis\n",
    "def get_sentiment(text):\n",
    "    return TextBlob(text).sentiment.polarity\n",
    "\n",
    "cleaned_reviews_df['Sentiment'] = cleaned_reviews_df['Text_Review'].apply(get_sentiment)\n",
    "cleaned_reviews_df['Sentiment_Category'] = pd.cut(cleaned_reviews_df['Sentiment'], bins=[-1, -0.1, 0.1, 1], labels=['Negative', 'Neutral', 'Positive'])\n",
    "\n",
    "# Sidebar\n",
    "st.sidebar.title(\"Review Analysis\")\n",
    "option = st.sidebar.selectbox(\"Choose Analysis\", [\"Overview\", \"Word Clouds\", \"Detailed Review\"])\n",
    "\n",
    "# Overview\n",
    "if option == \"Overview\":\n",
    "    st.title(\"Overview of Reviews\")\n",
    "    \n",
    "    st.subheader(\"Distribution of Ratings\")\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    sns.countplot(x='Rating', data=cleaned_reviews_df, palette='viridis')\n",
    "    plt.title('Distribution of Ratings')\n",
    "    plt.xlabel('Rating')\n",
    "    plt.ylabel('Count')\n",
    "    st.pyplot(plt)\n",
    "    \n",
    "    st.subheader(\"Distribution of Sentiment Categories\")\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    sns.countplot(x='Sentiment_Category', data=cleaned_reviews_df, palette='viridis')\n",
    "    plt.title('Distribution of Sentiment Categories')\n",
    "    plt.xlabel('Sentiment Category')\n",
    "    plt.ylabel('Count')\n",
    "    st.pyplot(plt)\n",
    "\n",
    "# Word Clouds\n",
    "elif option == \"Word Clouds\":\n",
    "    st.title(\"Word Clouds of Reviews\")\n",
    "    \n",
    "    positive_reviews = ' '.join(cleaned_reviews_df[cleaned_reviews_df['Sentiment_Category'] == 'Positive']['Text_Review'])\n",
    "    negative_reviews = ' '.join(cleaned_reviews_df[cleaned_reviews_df['Sentiment_Category'] == 'Negative']['Text_Review'])\n",
    "    \n",
    "    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)\n",
    "    wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)\n",
    "    \n",
    "    st.subheader(\"Word Cloud for Positive Reviews\")\n",
    "    plt.figure(figsize=(16, 8))\n",
    "    plt.imshow(wordcloud_positive, interpolation='bilinear')\n",
    "    plt.axis('off')\n",
    "    st.pyplot(plt)\n",
    "    \n",
    "    st.subheader(\"Word Cloud for Negative Reviews\")\n",
    "    plt.figure(figsize=(16, 8))\n",
    "    plt.imshow(wordcloud_negative, interpolation='bilinear')\n",
    "    plt.axis('off')\n",
    "    st.pyplot(plt)\n",
    "\n",
    "# Detailed Review\n",
    "elif option == \"Detailed Review\":\n",
    "    st.title(\"Detailed Review Analysis\")\n",
    "    \n",
    "    custom_stopwords = set([\n",
    "        'the', 'and', 'i', 'it', 'a', 'is', 'to', 'this', 'in', 'was', 'of', 'but', 'not', 'on', 'for', 'so', 'my', 'like', 'with', 'that', 'they', 'you', 'as', 'at', 'be'\n",
    "    ])\n",
    "    \n",
    "    def get_common_words(text):\n",
    "        words = text.split()\n",
    "        filtered_words = [word for word in words if word.lower() not in custom_stopwords]\n",
    "        word_counts = Counter(filtered_words)\n",
    "        return word_counts.most_common(20)\n",
    "    \n",
    "    st.subheader(\"Common Words in Negative Reviews\")\n",
    "    negative_words = ' '.join(cleaned_reviews_df[cleaned_reviews_df['Sentiment_Category'] == 'Negative']['Text_Review'])\n",
    "    common_negative_words = get_common_words(negative_words)\n",
    "    st.write(common_negative_words)\n",
    "    \n",
    "    st.subheader(\"Common Words in Positive Reviews\")\n",
    "    positive_words = ' '.join(cleaned_reviews_df[cleaned_reviews_df['Sentiment_Category'] == 'Positive']['Text_Review'])\n",
    "    common_positive_words = get_common_words(positive_words)\n",
    "    st.write(common_positive_words)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
