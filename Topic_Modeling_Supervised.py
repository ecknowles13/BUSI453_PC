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
    #from nltk.stem import PorterStemmer
    # extra import statements
    import numpy as np
    from collections import Counter

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
    # remove stopwords with lambda x
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

    # create function to stem
    # create stem object
    #porter = PorterStemmer()
    #def stem_sentence(sentence):
       # token_words = word_tokenize(sentence)
        #token_words
       # stem_sentence = []
    # create function to convert sentence to words
    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

    # function to remove stopwords
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    # clean text to a list
    data = article.Clean_text.values.tolist()
    # use function to break sentences into words
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

    # view the first row, first column, up to 30 words
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

    #####################
    # supervised use of topic modeling
    # dictionary of topics
    topics_dct = {
        'Topic_1': ['president', 'trump', 'city', 'york', 'biden', 'election'],
        'Topic_2': ['people', 'census', 'state', 'country', 'nation'],
        'Topic_3': ['market', 'vaccine', 'coronavirus', 'close', 'economy']
    }

    # create a clean copy of data set with only the columns that we want
    kw_df = article[['headline', 'abstract', 'Clean_text']].copy()
    # get the function for assigning topics from GitHub
    def generate_labels(topics_dct: dict, df: pd.DataFrame, col: str) -> pd.DataFrame:
        '''
        This method will generate binary labels associated with each topic in topics_dct
        by counting the keyword occurrences associated with that topic.

        Args:
            df: DataFrame holding the text column you want to scan over
            col: String corresponding to a column name in the input df
            topics_dct: Dictionary of unique topics and list of keywords you want to count
                        in the column

        Returns:
            The input dataframe with additional columns corresponding to the label for each
            topic in topics_dct
        '''
        for topic, kw in topics_dct.items():
            # count the occurrences of. the keywords in the summary
            df[topic + '_label'] = df[col].apply(
                lambda x: [1 for word, count in Counter(x.split()).items() if word in kw] or np.nan)

            # generate binary classification -- 1 if the keyword was in the article, 0 otherwise
            df[topic + '_label'] = df[topic + '_label'].apply(lambda x: [1 if isinstance(x, list) else np.nan][0])
            df[topic + '_label'] = df[topic + '_label'].fillna(0)
        return df


    kw_df = generate_labels(topics_dct, kw_df, 'Clean_text')

    print(kw_df.head().to_string())
    for col in ['Topic_1_label', 'Topic_2_label', 'Topic_3_label']:
        print(col, kw_df[col].sum())

    # take the labels generated and create a new data frame
    df_labels = kw_df[['headline', 'Topic_1_label', 'Topic_2_label', 'Topic_3_label']].copy()
    print(df_labels.head().to_string())

    # join back together with another data set based on a key (primary key)
    # headline is our key - occurs in both data sets
    complete_df = pd.merge(article, df_labels, on='headline', how='left')
    print(complete_df.head().to_string())
