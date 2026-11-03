import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def preparing_data(clean_html_urls=False):
    ## Load datasets
    df1 = pd.read_csv("./data/raw/emails.csv")
    df2 = pd.read_csv("./data/raw/spam_ham_dataset.csv")

    ## Cleaning and harmonizing datasets
    df1['text'] = df1['text'].str.replace(r'^subject:\s*', '', case=False, regex=True)
    df1["text"] = df1["text"].str.strip()

    df2["text"] = df2["subject"].fillna("") + " " + df2["body"].fillna("")
    df2["spam"] = df2["label"].map({"ham": 0, "spam": 1})
    df2.drop(columns=["source_folder", "filename", "subject", "body", "label"], inplace=True)
    df2["text"] = df2["text"].str.strip()

    df = pd.concat([df1, df2], ignore_index=True)

    # EXPLORATORY  DATA ANALYSIS (EDA)

    ## Calculating frequency of ham and spam emails
    print('\nDuplicate entries and null values before cleaning:')
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values}")
    duplicate_entries = df.duplicated().sum()
    print(f"Duplicate entries: {duplicate_entries}")

    ## Remove duplicate entries and missing values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    print('\nDuplicate entries and null values after cleaning:')
    duplicate_entries = df.duplicated().sum()
    print(f"Duplicate entries: {duplicate_entries}")
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values}")

    spam_count = df['spam'].value_counts()
    spam_perc = df['spam'].value_counts(normalize=True) * 100
    print(f"Ham emails: {spam_count[0]} ({spam_perc[0]:.2f}%), Spam emails: {spam_count[1]} ({spam_perc[1]:.2f})%. Total emails: {len(df)}")

    ## Cleaning HTML and URLs traces from the email text
    def cleaning_html_and_urls(text):
        text = re.sub(r'<style[^>]*>[\s\S]*?</style>', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'<script[^>]*>[\s\S]*?</script>', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'&\w+;', ' ', text)
        text = re.sub(r'\b[0-9a-f]{6}\b', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\b', ' ', text)
        return text
    if clean_html_urls:
        df['text'] = df['text'].apply(cleaning_html_and_urls)

    ## Creating a list of stop words including common HTML tags and attributes
    words_html = [
        'html', 'div', 'style', 'bgcolor', 'cellpadding', 'cellspacing',
        'enron', 'ect', 'vince', 'hou', 'zdnet', 'cnet', 'clickthru',
        'subject', 'list', 'net', 'com', 'font', 'br', 'td', 'tr', 'size', 'color', 'face', 'nbsp', 
        'http', 'com', 'width', 'arial', 'align', 'center', 'www', 
        'table', 'height', 'border', 'href', 'src', 'gif', 'img', 'ffffff', 
        'sans', 'seriff', 'verdana', 'helvetica', 'family', 'font-size', 'weight', 'style', '000'
    ]

    personalised_stop_words = list(sk.feature_extraction.text.ENGLISH_STOP_WORDS.union(words_html))

    def plot_top_words(axes, df, top_n = 20, stop_words = None, title = ""):

        vectorizer = CountVectorizer(stop_words=stop_words)
        X = vectorizer.fit_transform(df['text'])
        feature_names = vectorizer.get_feature_names_out()
        word_count = np.asarray(X.sum(axis=0)).flatten()
        word_freq = dict(zip(feature_names, word_count))
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = top_words[:top_n]

        sns.barplot(ax = axes, 
                    x=[word[1] for word in top_words],
                    y=[word[0] for word in top_words],
                    hue=[word[0] for word in top_words],
                    palette='Spectral')
        axes.set_title(title)
        axes.set_xlabel('Frequency')
        axes.set_ylabel('Words')



    spam_emails = df[df['spam'] == 1]
    ham_emails = df[df['spam'] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_top_words(axes[0], spam_emails, top_n=15, stop_words=personalised_stop_words, title="Top 20 Most Frequent Words in Spam emails")
    plot_top_words(axes[1], ham_emails , top_n=15, stop_words=personalised_stop_words, title="Top 20 Most Frequent Words in Ham emails")
    plt.tight_layout()
    plt.show()

    ## Dividing the dataset into training and testing sets
    X = df['text']
    y = df['spam']  

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return df, personalised_stop_words, X_train, X_test, y_train, y_test