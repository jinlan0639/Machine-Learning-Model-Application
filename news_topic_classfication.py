# -*- coding: utf-8 -*-
"""
@author: RaoJinLan
"""

# ========================================================#
# 1.import packages
# ========================================================#
import itertools
import string
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

os.chdir(r"D:\KCL\sem2 big data and text\course work")
os.getcwd()

default_stopwords = nltk.corpus.stopwords.words('english')
print(len(default_stopwords), default_stopwords)

# %%
# ========================================================#
# 2.data import and data explore
# ========================================================#

# import data
data0 = pd.read_csv("data_2022-3\second_part\Huff_news.csv", parse_dates=['date'], index_col=0)

# check data info
data0.info()

# check for data duplications for all columns
print(data0[data0.duplicated()].shape)

# check for data duplication for certain columns
print(data0.authors.value_counts().head(10))
print(data0.headline.value_counts().head(10))
print(data0.short_description.value_counts().head(10))

# check for the duplicated reason
print(data0[data0.headline.duplicated()].sort_values(by="headline"))
print(data0[data0.short_description.duplicated()].sort_values(by="short_description"))

# the category info
data0.category.value_counts().plot(kind="barh", figsize=(5, 5))
plt.title("Figure2:Category Distribution")
plt.show()


# the word count of authors/headline/short_description column
def word_count(x):
    ''' x : Series'''
    return x.str.split().str.len().value_counts().sort_index()


# word count visulization
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0][0].plot(word_count(data0.authors))
axs[0][0].set_title('1.words couunt of Authors')

axs[0][1].plot(word_count(data0.headline))
axs[0][1].set_title('2.words count of headlines')

axs[1][0].plot(word_count(data0.short_description))
axs[1][0].set_title('3.words count of short_description')

axs[1][1].plot(data0.date.value_counts().sort_index())
axs[1][1].set_title('4.time distribution')

plt.show()

'''
Our findings:
1. There are Nan Value in some columns.
2. There are duplicated data in 'authors' column ,which indicate some author could write more article. 
   There are duplicated data in 'headlines/short_description' columns.But it might be normal.
3. The number of categroies is quite different, so the sample is an uneven sample.
4. The 'authors/headline/short_description' include some info helpful for classification.
5. The date of news is distributed between May 2014 to Jan 2015.

Our approach:
1. Merge'authors/headline/short_description' as 'Text all'
2. Keep 'Text all/Catorgory' colmns and drop duplicates to form a new dataset data
'''

# %%
# ========================================================#
# 3.merge data,data pre-processing,EDA
# ========================================================#

# merge the data
data0['text_all'] = data0['authors'].fillna('') + data0['headline'].fillna('') + data0['short_description'].fillna('')

# drop duplicates
data = data0[['text_all', 'category']].drop_duplicates()

data.info()
# #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   text_all  24934 non-null  object
#  1   category  24934 non-null  object

# data pre-pcocessing
lemma = WordNetLemmatizer()
porter_stemmer = PorterStemmer()


def clean_text(doc,
               rm_punctuation=True,
               rm_digits=True,
               lemmatize=False,
               norm_case=True,
               stem=False,
               rm_stopwords=True):
    # Doc overall operations
    if (rm_digits == True):
        table = str.maketrans({key: None for key in string.digits})
        doc = str(doc).translate(table)
    if (norm_case == True):
        doc = doc.lower()
    if (rm_punctuation == True):
        table = str.maketrans({key: None for key in string.punctuation})
        doc = doc.translate(table)
    if (rm_stopwords == True):
        words = " ".join([i for i in doc.split() if i not in default_stopwords])
    else:
        words = " ".join([i for i in doc.split()])
    if (lemmatize == True):
        words = " ".join(lemma.lemmatize(word) for word in words.split())
    if (stem == True):
        words = " ".join(porter_stemmer.stem(word) for word in words.split())
    return words


data['text_all'] = [clean_text(x, stem=True, lemmatize=True) for x in data.text_all]


# ========================================================#
# EDA
# ========================================================#
# word cloud
def wordcloud_fn(x):
    from wordcloud import WordCloud, STOPWORDS
    stopwords1 = set(STOPWORDS)
    words = x.str.cat(sep='').lower()
    wordcloud = WordCloud(background_color='white', stopwords=stopwords1, min_font_size=10).generate(words)
    return wordcloud


fig, axes = plt.subplots(7, 4, figsize=(15, 15))
plt.subplots_adjust(hspace=2, wspace=2)

axes[0, 0].set_title(f"Wordcloud category: All Category", fontsize=12)
axes[0, 0].set_axis_off()
axes[0, 0].imshow(wordcloud_fn(data.text_all))

for i, cat in enumerate(data.category.unique()[:]):
    row = (i + 1) // 4
    col = (i + 1) % 4
    axes[row, col].set_title(f"Wordcloud category: {cat}", fontsize=12)
    axes[row, col].set_axis_off()
    axes[row, col].imshow(wordcloud_fn(data[data.category == cat].text_all))

plt.suptitle(f"Wordclouds for categories", fontsize=20)
plt.tight_layout()
plt.show()

# word count
data['text_all'].str.split().str.len().value_counts().sort_index(ascending=False).plot(kind='barh', figsize=(10, 10))
plt.title('Figure2.the distribution of number of words in each document')
plt.show()

# %%
# ========================================================#
# 4&5.Data Split and Feature representation
# ========================================================#

X, y = data.text_all, data.category

tfidf_vectorizer = TfidfVectorizer(max_features=3000, binary=False, max_df=0.8, min_df=5, stop_words=default_stopwords,
                                   ngram_range=(1, 3))
# binary_vectorizer = CountVectorizer(max_features=1000,max_df=0.8, min_df=5,stop_words=default_stopwords,ngram_range=(1,3),binary=True)
# count_vectorizer  = CountVectorizer(max_features=1000,max_df=0.8, min_df=5,stop_words=default_stopwords,ngram_range=(1,3))

X = tfidf_vectorizer.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# %%
# ========================================================#
# 6.train the model
# ========================================================#
#  LogitRegression()
lg = LogisticRegression(max_iter=1000)
lg.fit(X_train, y_train)
lg_prediction = lg.predict(X_test)

#  LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_prediction = lda.predict(X_test)

#  MultinomialNB()
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_prediction = nb.predict(X_test)

#  SVC()
SVC_model = SVC()
SVC_model.fit(X_train, y_train)
SVC_prediction = SVC_model.predict(X_test)

# Random forest()
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_prediction = rf_classifier.predict(X_test)

# %%
# ========================================================#
# 7.top 3 predictions acorrding to probabilities
# ========================================================#
# --lg
print((pd.Series(lg_prediction).value_counts() / len(lg_prediction)).sort_values(ascending=False).head(3))
# --lda
print((pd.Series(lda_prediction).value_counts() / len(lda_prediction)).sort_values(ascending=False).head(3))
# --nb
print((pd.Series(nb_prediction).value_counts() / len(nb_prediction)).sort_values(ascending=False).head(3))
# --nb
print((pd.Series(SVC_prediction).value_counts() / len(SVC_prediction)).sort_values(ascending=False).head(3))

# %%
# ========================================================#
# 8.evaluation metric
# ========================================================#

# accuracy
print("lg:", accuracy_score(lg_prediction, y_test))
print("lda:", accuracy_score(lda_prediction, y_test))
print("nb:", accuracy_score(nb_prediction, y_test))
print("svc:", accuracy_score(SVC_prediction, y_test))
print("rf:", accuracy_score(rf_prediction, y_test))

# precision_score
print("lg:", precision_score(lg_prediction, y_test, average='weighted'))
print("lda:", precision_score(lda_prediction, y_test, average='weighted'))
print("nb:", precision_score(nb_prediction, y_test, average='weighted'))
print("svc:", precision_score(SVC_prediction, y_test, average='weighted'))
print("rf:", precision_score(rf_prediction, y_test, average='weighted'))

# recall_score
print("lg:", recall_score(lg_prediction, y_test, average='weighted', zero_division=0))
print("lda:", recall_score(lda_prediction, y_test, average='weighted', zero_division=0))
print("nb:", recall_score(nb_prediction, y_test, average='weighted', zero_division=0))
print("svc:", recall_score(SVC_prediction, y_test, average='weighted', zero_division=0))
print("rf:", recall_score(rf_prediction, y_test, average='weighted', zero_division=0))

# f1_score
print("lg:", f1_score(lg_prediction, y_test, average='weighted'))
print("lda:", f1_score(lda_prediction, y_test, average='weighted'))
print("nb:", f1_score(nb_prediction, y_test, average='weighted'))
print("svc:", f1_score(SVC_prediction, y_test, average='weighted'))
print("rf:", f1_score(rf_prediction, y_test, average='weighted'))

# classfication report
print(classification_report(lg_prediction, y_test))
print(classification_report(lda_prediction, y_test))
print(classification_report(nb_prediction, y_test))
print(classification_report(SVC_prediction, y_test))
print(classification_report(rf_prediction, y_test))

# confusion matrix
cm_lg = confusion_matrix(y_test, lg_prediction)
cm_nb = confusion_matrix(y_test, nb_prediction)
cm_lda = confusion_matrix(y_test, lda_prediction)
cm_svc = confusion_matrix(y_test, SVC_prediction)
cm_rf = confusion_matrix(y_test, rf_prediction)

# confusion matrix visulization
fig, axes = plt.subplots(2, 2, figsize=(20, 20))

sns.heatmap(cm_lg, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title(f'Logistic Regression')

sns.heatmap(cm_lda, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title(f'LDA')

sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title(f'Naive Bayes')

sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title(f'SVC')

plt.tight_layout()
plt.show()