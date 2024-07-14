# importing necessary packages
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from bs4 import BeautifulSoup
import requests

# ################################################################################################## #
# #######################                 Preprocessing                 ############################ #
# ################################################################################################## #
# if you want to run this code, bring "True.csv" and "Fake.csv" files out of the base folder to the main directory and then run it.

"""
# reading initial data from csv #
full_true_df = pd.read_csv('True.csv')
full_fake_df = pd.read_csv('Fake.csv')

# assigning targets #
full_true_df['is_fake'] = 0
full_fake_df['is_fake'] = 1

# merging dataframes together and shuffling the final dataframe #
full_df = pd.concat([full_true_df, full_fake_df])
full_df = full_df.sample(frac=1).reset_index(drop=True)

# dropping columns #
full_df.drop(['date', 'title', 'subject'], axis=1, inplace=True)

# changing to lowercase #
full_df['text'] = full_df['text'].apply(lambda text: text.lower())

# dropping duplicates #
full_df.drop_duplicates(inplace=True)

# removing punctuations #
punctuation = string.punctuation


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punctuation))  # returns every character unless it's in punctuation


full_df["text"] = full_df["text"].apply(lambda text: remove_punctuation(text))

# removing stopwords #
# nltk.download('stopwords') <run this only once to download stopwords>
STOPWORDS = set(stopwords.words('english'))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


full_df["text"] = full_df["text"].apply(lambda text: remove_stopwords(text))

# making new csv file
full_df.to_csv('final_data.csv', index=False)
"""

# ################################################################################################## #
# #######################                 classification                ############################ #
# ################################################################################################## #

# reading from new csv file and making sure it is ok
df = pd.read_csv('final_data.csv')

# print(df.info())  # it appears that we have a few missed cells, we should remove them

df.dropna(inplace=True)

# print(df.info())  # it is fixed now

# splitting train and test data #

x = df['text']
y = df['is_fake']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# making classificator with pipeline #

vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')

mnb = MultinomialNB()  # /a bit slow, 95% accurate
# knn = KNeighborsClassifier(n_neighbors=3) /fast, 70% accurate
# dtree = DecisionTreeClassifier() /super slow, 99% accurate

clf = make_pipeline(vectorizer, mnb)
clf.fit(x_train, y_train)

# print(clf.score(x_test, y_test))

# cross validation #
# cross_val_scores = cross_val_score(clf, x, y, cv=5)
# print(cross_val_scores)
# print(cross_val_scores.mean())
# print(cross_val_scores.std())

# ################################################################################################## #
# #########################                 Web Scraping                ############################ #
# ################################################################################################## #

# using request to access cnn
url = "https://edition.cnn.com/"
result = requests.get(url)

# using beautifulsoup to scrape the data
doc = BeautifulSoup(result.text, "html.parser")
news = doc.find_all(class_="container__headline-text")

# making a dataframe of the data
new_X = []

for item in news:
    new_X.append(item.text)

new_df = pd.DataFrame(new_X, columns=["text"])

# preparing data for testing
punctuation = string.punctuation


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punctuation))


STOPWORDS = set(stopwords.words('english'))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


new_df["text"] = new_df["text"].apply(lambda text: remove_stopwords(remove_punctuation(text)))

# testing
test_target = list(clf.predict(new_df["text"]))
print(test_target)

new_df["is_fake"] = test_target
