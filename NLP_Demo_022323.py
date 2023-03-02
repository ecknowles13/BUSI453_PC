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

    # create a function to convert sentence to words
    def sent_to_word(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) # deacc = True then it removes punctuation

    # function that removes stopwords
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    data = article.Clean_text.values.tolist()
    data_words = list(sent_to_word(data))
    # use function to remove the stopwords
    data_words = remove_stopwords(data_words)
    print(data_words[:1][0][:20])

    ################## Topic Modeling
    # create a dictionary
    id2word = corpora.Dictionary(data_words)
    # create corpus
    texts = data_words
    # term document frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # view
    print(corpus[:1][0][:20])

    # LDA model training
    # keep the default parameters to keep it simple, but will check the number of topics
    num_tops = 20

    # build the LDA model
    lda_model = gensim.models.LdaMulticore(corpus = corpus, id2word=id2word, num_topics=num_tops)

    # print the keyword for each topic
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # visualize the topics
    visualization = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(visualization, 'LDA_Visualization.html')

    # Sentiment analysis using TextBlob
    # create a column for polarity, by using apply to iterate across data rows
    article['polarity'] = article.apply(lambda x: TextBlob(x['Clean_text']).sentiment.polarity, axis = 1)
    # display
    print(article.head())

    # sentiment analyzer using Vader
    # import nltk vader_lexicon
    ntlk.downloader.download('vader_lexicon')
    # create sentiment intensity analyzer object
    sid = SentimentIntensityAnalyzer()

    # apply across
    article['sentiments'] = article['Clean_text'].apply(lambda x: sid.polarity_scores(x))
    article = pd.concat([article.drop(['sentiments'], axis = 1), article['sentiments'].apply(pd.Series)], axis = 1)

    # view new dataframe
    print(article.head())

    # export to excel
    article.to_excel('Article_Sent_Analysis.xlsx')