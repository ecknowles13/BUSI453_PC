import pandas as pd
import re # cleansing with regular expression library
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import gensim
from gensim.utils import simple_preprocess
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint
import pyLDAvis.gensim_models
import pyLDAvis
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# load data
moto = pd.read_csv('/Users/emilyknowles/Downloads/Moto_Reviews.csv')
print(moto.head(5))

# data cleansing
# filter for reviews of 3
moto_3 = moto[moto['Rating'] == 3]
print(moto_3.head())

# remove columns, just reviews
moto_3 = moto_3.drop(columns = ['Rating', 'Comment'], axis = 1)
print(moto_3.head())

# remove punctuation/lower case
moto_3['Clean_text'] = moto_3['Review_text'].map(lambda x: re.sub('[,.!?-]', '', x))
print(moto_3.head())
# convert to lowercase
moto_3['Clean_text'] = moto_3['Clean_text'].map(lambda x: x.lower())
print(moto_3.head())

# EDA
# join all text together in a single string
string = ','.join(list(moto_3['Clean_text'].values))
# create a word cloud object
wordcloud = WordCloud(background_color = 'white',
                          max_words = 500).generate(string)
# visualize
plt.imshow(wordcloud)
plt.axis("off")
# save fig
plt.savefig("Word Cloud 2.png")

# create stopwords object
stop_words = stopwords.words('english')
# add to stopwords
stop_words.extend(['more'])

# create function to convert sentence to words
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

# function to remove stopwords
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
data = moto_3.Clean_text.values.tolist()
data_words = list(sent_to_words(data))
# use function to remove stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])


############################################ Topic Modeling
# create dictionary
id2word = corpora.Dictionary(data_words)
# create corpus
texts = data_words
# term document frequency
corpus = [id2word.doc2bow(text) for text in texts]

# view
print(corpus[:1][0][:30])

