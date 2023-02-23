if __name__ == '__main__':
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
    #import openpyxl

    # load data
    article = pd.read_csv('C:/Users/ecriggins/Downloads/NYT_articles.csv')
    print(article.head(5).to_string())

    # remove columns, just reviews
    article = article[['headline', 'abstract']]
    print(article.head())
    print(article.dtypes)

    # remove punctuation/lower case
    article['Clean_text'] = article['abstract'].map(lambda x: re.sub('\W+', ' ', str(x)))
    print(article.head())
    # convert to lowercase
    article['Clean_text'] = article['Clean_text'].map(lambda x: x.lower())
    print(article.head())
    # remove words less than 3 characters
    article['Clean_text'] = article['Clean_text'].map(lambda x: re.sub(r'\b\w{1,3}\b', '', str(x)))
    print(article.head())
    # remove stopwords
    # create stopwords object
    stop_words = stopwords.words('english')
    # add to stopwords
    stop_words.extend(['pandemic'])
    # remove stopwords
    article['Clean_text'] = article['Clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # EDA
    # join all text together in a single string
    string = ','.join(list(article['Clean_text'].values))
    # create a word cloud object
    wordcloud = WordCloud(background_color='white', max_words=500).generate(string)
    # visualize
    plt.imshow(wordcloud)
    plt.axis("off")
    # save fig
    plt.savefig("Word Cloud Articles.png")

    # create function to convert sentence to words
    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

    # function to remove stopwords
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    data = article.Clean_text.values.tolist()
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

    # LDA model training
    # keeping default params to keep it simple, but will chx number of topics
    num_topics = 4

    # build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
    # print the keyword for each topic
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # Visualize the topics
    visualization = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(visualization, 'LDA_Visualization.html')

    # Sentiment analysis Using TextBlob
    # create a column for polarity, by using apply to iterate across data rows
    article['polarity'] = article.apply(lambda x: TextBlob(x['Clean_text']).sentiment.polarity, axis=1)
    # display
    print(article.head())

    # sentiment analyzer using Vader
    # import nltk vader_lexicon
    nltk.downloader.download('vader_lexicon')
    # create sentiment intensity analyzer object
    sid = SentimentIntensityAnalyzer()

    # apply
    article['sentiments'] = article['Clean_text'].apply(lambda x: sid.polarity_scores(x))
    article = pd.concat([article.drop(['sentiments'], axis=1),
                        article['sentiments'].apply(pd.Series)], axis=1)

    # view df
    print(article.head())

    # export to excel
    article.to_excel('Article_Sentiment_Analysis.xlsx')

