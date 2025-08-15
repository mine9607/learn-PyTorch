import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from utils import ensure_movie_csv

nltk.download("stopwords")


def load_movie_data(csv_path: Path = Path("movie_data.csv")) -> pd.DataFrame:
    csv_file = ensure_movie_csv(csv_path)
    return pd.read_csv(csv_file)


def data_info(df):
    print(df.info())


def analyze_data(df):
    print(df.describe())


# ## Use Regex to remove HTML characters
def preprocessor(text):
    # Remove HTML tags
    text = re.sub("<[^>]*>", "", text)
    # Extract emoticons
    emoticons = re.findall(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    # Remove non-word characters, lowercase, append emoticons without '-'
    text = (
        re.sub(r"[\W]+", " ", text.lower()) + " " + " ".join(emoticons).replace("-", "")
    )
    return text


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def tokenizer_plus(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = (
        re.sub(r"[\W]+", " ", text.lower()) + " " + " ".join(emoticons).replace("-", "")
    )
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, "r", encoding="utf-8") as csv:
        next(csv)  # skip the header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


# --- Main Script Entry ----
if __name__ == "__main__":
    df = load_movie_data()
    print(df.shape, df.head(3), sep="\n")

    data_info(df)

    # Transform words into feature vectors (BAG OF WORDS MODEL)
    count = CountVectorizer()
    docs = np.array(
        [
            "The sun is shining",
            "The weather is sweet",
            "The sun is shining, the weather is sweet, and one and one is two",
        ]
    )

    bag = count.fit_transform(docs)

    print(count.vocabulary_)
    print(sorted(count.vocabulary_))
    print(bag.toarray())

    # Down weight frequently occurring words in documents since they won't help descriminate between documents
    tfidf = TfidfTransformer(use_idf=True, norm="l2", smooth_idf=True)
    np.set_printoptions(precision=2)
    print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

    # Cleaning the Text Data (test)
    print(df.loc[0, "review"][-50:])
    print(preprocessor(df.loc[0, "review"][-50:]))

    # Apply the cleaning preprocessor to the entire dataframe
    df["review"] = df["review"].apply(preprocessor)

    # How to split the text corpora into individual elements.
    # tokenize documents by splitting on whitespace
    tokens = tokenizer("runners like running and thus they run")
    print(tokens)

    porter = PorterStemmer()
    porter_tokens = tokenizer_porter("runners like running and thus they run")
    print(porter_tokens)

    # Remove stop words
    stop = stopwords.words("english")
    cleaned_words = [
        w
        for w in tokenizer_porter("a runner likes running and runs a lot")
        if w not in stop
    ]

    print(cleaned_words)

    # Train a logistic regression classifier for document classification
    X_train = df.loc[:25000, "review"].values
    y_train = df.loc[:25000, "sentiment"].values
    X_test = df.loc[25000:, "review"].values
    y_test = df.loc[25000:, "sentiment"].values
    print(X_train.shape, X_test.shape)

    # Bettter approach to randomize data and stratify
    X = df.loc[:, "review"].values
    y = df.loc[:, "sentiment"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)
    print(X_train.shape, X_test.shape)

    # Use GridSearchCV to find the optimal set of parameters for the LR classifier
    n_folds = 5
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    small_param_grid = [
        {
            "vect__ngram_range": [(1, 1)],
            "vect__stop_words": [None],
            "vect__tokenizer": [tokenizer, tokenizer_porter],
            "clf__penalty": ["l2"],
            "clf__C": [1.0, 10.0],
        },
        {
            "vect__ngram_range": [(1, 1)],
            "vect__stop_words": [stop, None],
            "vect__tokenizer": [tokenizer],
            "vect__use_idf": [False],
            "vect__norm": [None],
            "clf__penalty": ["l2"],
            "clf__C": [1.0, 10.0],
        },
    ]

    lr_tfidf = Pipeline(
        [("vect", tfidf), ("clf", LogisticRegression(solver="liblinear"))]
    )

    # gs_lr_tfidf = GridSearchCV(
    #     lr_tfidf,
    #     small_param_grid,
    #     scoring="accuracy",
    #     cv=5,
    #     verbose=2,
    #     n_jobs=-1,
    # )
    #
    # gs_lr_tfidf.fit(X_train, y_train)
    #

    # Print best parameter set
    # print(f"Best parameter set: {gs_lr_tfidf.best_params_}")
    # print(f"CV Accuracy: {gs_lr_tfidf.best_score_:.3f}")
    # clf = gs_lr_tfidf.best_estimator_
    # print(f"Test Accuracy: {clf.score(X_test, y_test):.3f}")

    # Out of Core Learning

    # ## Test streaming
    print(next(stream_docs(path="movie_data.csv")))
    vect = HashingVectorizer(
        decode_error="ignore",
        n_features=2**21,
        preprocessor=None,
        tokenizer=tokenizer_plus,
    )

    clf = SGDClassifier(loss="log", random_state=1)
    doc_stream = stream_docs(path="movie_data.csv")

    classes = np.array([0, 1])
    for _ in tqdm(range(45), desc="Training batches"):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)

    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print(f"Accuracy: {clf.score(X_test, y_test):.3f}")

    clf = clf.partial_fit(X_test, y_test)

    # Topic Modeling with latent Dirichlet allocation

    df = pd.read_csv("movie_data.csv", encoding="utf-8")

    count = CountVectorizer(stop_words="english", max_df=0.1, max_features=5000)
    X = count.fit_transform(df["review"].values)

    lda = LatentDirichletAllocation(
        n_components=10, random_state=123, learning_method="batch"
    )

    X_topics = lda.fit_transform(X)

    print(lda.components_.shape)

    n_top_words = 5
    feature_names = count.get_feature_names_out()
    print("Feature Names: ", feature_names)
    for topic_idx, topic in enumerate(lda.components_):
        print(f"Topic {(topic_idx + 1)}:")
        print(
            " ".join(
                [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
            )
        )

    horror = X_topics[:, 5].argsort()[::-1]
    for iter_idx, movie_idx in enumerate(horror[:3]):
        print(f"\nHorror movie #{(iter_idx + 1)}:")
        print(df["review"][movie_idx][:300], "...")
