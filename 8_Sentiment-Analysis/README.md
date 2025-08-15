# Chapter 8: Applying Machine Learning to Sentiment Analysis

`Sentiment Analysis` is a subfield of `natural language processing`

- Cleaning and preparing text data
- Building feature vectors from text documents
- Training ML models to classify positive and negative movie reviews
- Working with large text datasets using out-of-core learning
- Inferring topics from document collections for categorization

## Preparing the Dataset

IMDB Dataset

positive means that a movie was rated higher than six stars on IMDb

negative means that a movie was rated lower than five stars on IMDb

### Preprocessing the Dataset

#### Prepare Dataframe with Features (X) and labels (y)

#### Shuffle the data

## Bag of Words Model

### Transform words into Feature Vectors (Vectorization)

#### Count Vectorization (Simple)

#### Assessing Word Relevancy: Term Frequency-Inverse Document Frequency (tf-idf)

### Cleaning text Data

#### Remove unwanted characters

### Process Documents into tokens

#### Split on Whitespace

#### Stemming

## Training a LogisticRegression Classifier for Document Classification

## Out-of-Core Learning (Streaming or Minibatch)

## Topic Modeling: Latent Dirichlet Allocation

`topic modeling` - describes the task of assigning topics to unlabeled text documents

In topic modeling we aim to assign category labels to documents in a corpus of documents (e.g.--assign newspaper articles to sports, finance, world news, politics, local news, etc.)

Topic modeling is considered a `clustering` task - `unsupervised learning`

`Latent Dirichlet Allocation (LDA)` - is a popular technique for topic modeling. It is a generative probabilistic model that tries to find groups of words that appear frequently together across different documents. These frequently appearing words represent our 'topics'. The input to an LDA is the `bag-of-words model`

##
