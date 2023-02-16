# import packages
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# load data
wordgame = pd.read_csv('~/Downloads/wordgame.csv')
print(wordgame.head().to_string())

# filter for a particular word, in this case radio
wordgame_radio = wordgame[wordgame['word1'] == 'radio']
print(wordgame_radio.head())

# remove columns, just word2 and author
wordgame_radio = wordgame_radio.drop(columns=['source', 'sourceID'], axis=1)
print(wordgame_radio.head())

# EDA
# join all text together in a single string
string = ','.join(list(wordgame_radio['word2'].values))

# bar chart
wordgame_radio['word2'].value_counts().plot(kind = "bar")
plt.xlabel("Word")
plt.xlim(-.5, 20.5)
plt.ylabel("Count")
plt.title("Word Association for Radio")
plt.show()

# create a word cloud object
wordcloud = WordCloud(background_color='white',
                      max_words=100).generate(string)
# visualize
plt.imshow(wordcloud)
plt.axis("off")
# save fig
plt.savefig("Word Cloud Radio.png")