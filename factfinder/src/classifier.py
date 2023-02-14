import re

import joblib
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stopwords = nltk.corpus.stopwords.words("russian")


def read_data(filepath="total_reports.xlsx"):
    # TODO: unify data reading code, create a dataframe structure template
    data = pd.read_excel(filepath)
    data = data[~data["Блок"].isin(["Не ЦУР", "БОС"])]
    data = data.dropna(subset=["Блок", "Текст"])
    data = data[["Блок", "Текст"]]
    data = data.rename(columns={"Блок": "y", "Текст": "text"})
    data["length"] = data.text.map(lambda x: len(x))
    return data


def preprocess(text, need_stem=False, need_lemmatize=True, lst_stopwords=None):
    text = re.sub(r"[^\w\s]", "", str(text).lower().strip())

    lst_text = text.split()
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    if need_stem:  # TODO: remove or replace stemmer
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    if need_lemmatize:  # TODO: replace lemmatizer
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    text = " ".join(lst_text)
    return text


def split_data(data):
    X, y = data.drop(columns="y"), data["y"]
    return train_test_split(
        X, y, test_size=0.4, random_state=42
    )  # random_state is required!


def train_pipeline(X_data, y_data):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    classifier = MultinomialNB()
    model = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )
    model.fit(X_data, y_data)
    return model


def save_model(model, model_path="model_classifier.joblib"):
    joblib.dump(model, model_path)


if __name__ == "__main__":
    df = read_data()
    df["text_clean"] = df["text"].apply(
        lambda x: preprocess(x, need_lemmatize=True, lst_stopwords=stopwords)
    )
    X_train, X_test, y_train, y_test = split_data(df[["clean_text", "y"]])
    pipeline = train_pipeline(X_train, y_train)
    save_model(pipeline)
    # TODO: quality report creation
