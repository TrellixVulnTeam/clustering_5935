from sklearn.feature_extraction.text import TfidfVectorizer
import math
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import string
import numpy

nltk.download('punkt')  # first-time use only
stemmer = nltk.stem.porter.PorterStemmer()


# Stemming

def StemTokens(tokens):
    return [stemmer.stem(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def StemNormalize(text):
    return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Lemmatization

nltk.download('wordnet')  # first-time use only
lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def distance(str1, str2):

    LemVectorizer = CountVectorizer(
        tokenizer=LemNormalize, stop_words='english')
    LemVectorizer.fit_transform([str1, str2])
    print(LemVectorizer.vocabulary_)
    tf_matrix = LemVectorizer.transform([str1, str2]).toarray()
    tfidfTran = TfidfTransformer(norm="l2")
    tfidfTran.fit(tf_matrix)
    tfidf_matrix = tfidfTran.transform(tf_matrix)
    cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    return cos_similarity_matrix[0, 1]


print(distance("Hello girl", "Hello boy"))


# TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')


# def cos_similarity(textlist):
#     tfidf = TfidfVec.fit_transform(textlist)
#     return (tfidf * tfidf.T).toarray()


# print(cos_similarity(["Hello girl", "Hello boy"]))
