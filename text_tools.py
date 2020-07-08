import pandas
from sklearn.feature_extraction.text import TfidfVectorizer


def similarity(str1, str2):
    # print("Here 1")
    vect = TfidfVectorizer(min_df=1, stop_words="english")
    tfidf = vect.fit_transform([str1, str2])
    pairwise_similarity = tfidf * tfidf.T
    # print(pairwise_similarity[0, 1])
    # print("Here 2")
    return pairwise_similarity[0, 1]

def distance(x, y):

    str1 = x.iloc[0]
    str2 = y.iloc[0]
    # print("X :", str1, "Y", str2)
    if str1 != str2:
        return 10

    return 10 - 10 * similarity(str1=x['Message'].astype(str), str2=y['Message'].astype(str))


def distance2(type_1, type_2, message_1, message_2):

    if isinstance(type_2, str):
        print("Type 2 is string ")
        if similarity(str1=type_1, str2=type_2) < 0.5:
            print("Differente types")
            return 10
        print("Distance est : ", 10 - 10 * similarity(str1=message_1, str2=message_2))
        return 10 - 10 * similarity(str1=message_1, str2=message_2)
    if isinstance(type_2, pandas.core.series.Series):
        print("Type 2 is series ")
        print("Type 2 content ", type_2)
        print("Type 2 Type ", type_2.iloc[0])
        if similarity(str1=type_1, str2=type_2.iloc[0]) < 0.3:
            print("Differente types")
            return 10
        print("Distance series est : ", 10 - 10 * similarity(str1=message_1, str2=message_2.iloc[0]))
        return 10 - 10 * similarity(str1=message_1, str2=message_2.iloc[0])

