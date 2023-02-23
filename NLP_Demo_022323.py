if __name__ == '__main__':
    import pandas as pd
    import re
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    import gensim
    from gensim.utils import simple_preprocess
    import nltk

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    import gensim.corpora as corpora
    from pprint import pprint
    import pyLDAvis
    import pyLDAvis.gensim_models
    from textblob import TextBlob
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # load data
    article = pd.read_csv('C:/Users/ecriggins/Downloads/NYT_articles.csv')
    print(article.head().to_string())

    # remove columns we don't need
    article = article[['headline', 'abstract']]
    print(article.head().to_string())

    # remove punctuation and make it lowercase
    article['Clean_text'] = article['abstract'].map(lambda x: re.sub('\W+', ' ', str(x)))
    print(article.head().to_string())

    # convert to lowercase
    article['Clean_text'] = article['Clean_text'].map(lambda x: x.lower())
    print(article.head().to_string())

    # remove words of less than 3 characters
    article['Clean_text'] = article['Clean_text'].map(lambda x: re.sub(r'\bw{1,3}\b', '', str(x)))
    print(article.head())

    # remove stopwords
    # create a stopwords object
    stop_words = stopwords.words('english')
    # add to our stopwords library
    stop_words.extend(['covid', 'pandemic', 'economy'])
    # remove stopwords
    article['Clean_text'] = article['Clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # visualize
    # join all the text together in a string
    string = ','.join(list(article['Clean_text'].values))
    # create a wordcloud object
    wordcloud = WordCloud(background_color='white', max_words=120).generate(string)
    plt.imshow(wordcloud)
    plt.axis("off")
    # save figure
    plt.savefig("NYT Word Cloud.png")