# importing needed packages
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# load data
wordgame = pd.read_csv("C:/Users/ecriggins/Downloads/wordgame.csv")
print(wordgame.head().to_string())

print(wordgame.shape)

# filter for a particular word - this case, radio
prompt_radio = wordgame[wordgame['word1'] == 'radio']
print(prompt_radio.head(30))

# drop extraneous columns from data frame
prompt_radio = prompt_radio.drop(columns = ['source', 'sourceID'], axis=1)
print(prompt_radio.head())

# join all the text together in a single string
string = ','.join(list(prompt_radio['word2'].values))

print(string)

# create a bar chart
prompt_radio['word2'].value_counts().plot(kind = "bar")
plt.xlabel("Word Response")
plt.ylabel("Count")
plt.title("Frequency of Word Association - Radio")
plt.xlim(-.5, 20.5)
plt.show()
plt.savefig("Bar Chart - Radio.png")
# create a word cloud object
wc = WordCloud(background_color = 'white', max_words = 100).generate(string)

# visualize
plt.imshow(wc)
plt.axis("off")

# save figure
plt.savefig("Word Cloud Radio.png")
