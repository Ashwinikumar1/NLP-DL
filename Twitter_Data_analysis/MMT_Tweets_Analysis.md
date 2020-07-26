
#### Objective : Recently, i have came across lot of posts regarding MMT not refunding or delaying refunds of customer cancellation. I suspect a lot of customers facing this issue during Lockdown. The objective of this analysis is to undertsand the extent of problem and see what people are complaining about
#### Data Source : We will extract tweets since march using tweepy api which have make my trip in them and analyze them. Mainly we will do simple sentiment analysis and topic clustering 
#### Date : 26th Jul 2020
#### Author : Ashwini Kumar


```python
#### Import the required packages 
import os ### for all os functions
import tweepy as tw ### Tweepy is a api which help us interact with 
import pandas as pd ### This will help us to manipulate and do data wrangling on pandas dataframe
import pickle ### To serialize and store python object for later use
import spacy ### for text preprocessing
import nltk ### for sentiment analysis
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib import pyplot as plt
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

from nltk import tokenize
```


```python
#### As we are using twitter api, we need developer access to twitter and other privileges. It is easy to get aceess by following step by step guide
#### The link for this is https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/twitter-data-in-python/
consumer_key= 'jZZaXN8rDN5Mo27QION9jtrfQ'
consumer_secret= 'MMh0GA0qtGQnqMtLVuQMFtU71G0SLHUTLqCYeAsOuKJxS1erDO'
access_token= '1287091027354910720-3lRHdkD6Sdli8W351ILJ1AAd3Y5wWi'
access_token_secret= '00YDniRn2leOaaAlP12Eym6sOjfIsuU8SasRH7SC5K49Q'
```


```python
#### Pass all the authetication handler to be used and set up the access tojen
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret) ### Pass the access token to be used 
api = tw.API(auth, wait_on_rate_limit=True) ###  set true = automatically wait for rate limits to replenish
```


```python
#### Define the function whch searchs for all the keywords for a given data and return a pandas dataframe

def get_tweets(search_words,language,date_since,mode,tweet_count):
    '''
    Define a get tweets function which search the tweets for a keyword of a list of keywords and return defined number of tweets
    search_word : Keyword to be searched. It can be a string or list of strings
    language : The language of keywords to be searched 
    date_since : The date from which keywords should be searched
    tweet_count : Define extended otherwise tweets are trimmed to 140 characters and rest is provided with https
    items : Number of tweets to be searched. If you want all define it to be a large number
    Return : Its return a dataframe with the tweets as required
    Usage:
    
    '''
    tweets = tw.Cursor(api.search, search_words,lang=language, since=date_since,tweet_mode=mode).items(tweet_count)
    tweets_list = []
    for count,i in enumerate(tweets):
        tweets_list.append([i.full_text,i.created_at,i.id,i.retweeted])
        if count % 100 == 0:
            print ("Found tweets with keywords :", count)
    
    print ("Total tweets found :",count)
    return (pd.DataFrame(tweets_list, columns = ['tweets_text','created_time_stamp','tweet_id','retweet_flag']))
```


```python
### Call the above function and store all the tweets in a dataframe
tweets_df_temp = get_tweets("makemytrip","en","2020-03-03","extended",10000000)
```

    Found tweets with keywords : 0
    Found tweets with keywords : 100
    Found tweets with keywords : 200
    Found tweets with keywords : 300
    Found tweets with keywords : 400
    Found tweets with keywords : 500
    Found tweets with keywords : 600
    Found tweets with keywords : 700
    Found tweets with keywords : 800
    Found tweets with keywords : 900
    Found tweets with keywords : 1000
    Found tweets with keywords : 1100
    Found tweets with keywords : 1200
    Found tweets with keywords : 1300
    Found tweets with keywords : 1400
    Found tweets with keywords : 1500
    Found tweets with keywords : 1600
    Found tweets with keywords : 1700
    Found tweets with keywords : 1800
    Found tweets with keywords : 1900
    Found tweets with keywords : 2000
    Found tweets with keywords : 2100
    Found tweets with keywords : 2200
    Found tweets with keywords : 2300
    Found tweets with keywords : 2400
    Found tweets with keywords : 2500
    Found tweets with keywords : 2600
    Found tweets with keywords : 2700
    Found tweets with keywords : 2800
    Found tweets with keywords : 2900
    Found tweets with keywords : 3000
    Found tweets with keywords : 3100
    Found tweets with keywords : 3200
    Found tweets with keywords : 3300
    Found tweets with keywords : 3400
    Found tweets with keywords : 3500
    Found tweets with keywords : 3600
    Total tweets found : 3622
    


```python
### Pickle the dataframe , as pulling is time consuming process. So that we can directly use it
pickle.dump(tweets_df_temp,open("mmt_tweets.pickle",'wb'))
### Drop the tweets which have same text in tweets
print ("Shape of data before dropping duplicate tweets text is :",tweets_df_temp.shape )
tweets_df = tweets_df_temp.drop_duplicates(subset = ['tweets_text']) ### only consider full text while dropping
print ("Shape of data after dropping duplicate tweets text is :",tweets_df.shape )
```

    Shape of data before dropping duplicate tweets text is : (3623, 4)
    Shape of data after dropping duplicate tweets text is : (3223, 4)
    


```python
### Lets have a look at final data
tweets_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweets_text</th>
      <th>created_time_stamp</th>
      <th>tweet_id</th>
      <th>retweet_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@makemytripcare @makemytrip Hello Team\nI am a...</td>
      <td>2020-07-26 07:07:29</td>
      <td>1287283380103544832</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@Mr_ankur_anand @makemytrip Hi Ankur Anan. We ...</td>
      <td>2020-07-26 07:06:44</td>
      <td>1287283193494872064</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@AirAsiaSupport @malindoair  Our travel for Au...</td>
      <td>2020-07-26 07:03:18</td>
      <td>1287282327392051201</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@thedoubletdiary @IndiGo6E @DGCAIndia @makemyt...</td>
      <td>2020-07-26 07:02:06</td>
      <td>1287282025531977728</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RT @Shraddh77591427: We don't want Bollywood j...</td>
      <td>2020-07-26 07:01:20</td>
      <td>1287281835760787456</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Look at single text
tweets_df['tweets_text'][0]
```




    '@makemytripcare @makemytrip Hello Team\nI am asked to wait for 15 days. Indigo sent a mail on 2nd April for the issuance of credit shell. I have to book flight tomorrow. why I should wait for 15 days, its not fair. we payed the fee to MMT as well for the booking and we need the support. please help.'




```python
### Look at the length of texts
%matplotlib inline
tweets_df['tweets_text'].apply(len).plot(kind = 'hist',title ='Histogram By length of tweets text')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11d246a0>




![png](MMT_Tweets_Analysis_files/MMT_Tweets_Analysis_9_1.png)



```python
### Lets do some basic data cleaning i.e. remove https and puntuations
def basic_clean(text):
    '''Use regex to clean he text as required'''
    url = re.compile(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)")
    return url.sub(r'',text)
```


```python
tweets_df['tweets_text_cleaned'] = tweets_df['tweets_text'].apply(basic_clean)
```


```python
#### Create a Word Cloud for the analysis
text = " ".join(review.lower() for review in tweets_df['tweets_text_cleaned'])
stopwords = set(STOPWORDS)
stopwords.update(['makemytrip','mmt','please','will'])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
print ("Producing Word Cloud ")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show() 
# path = "C:\\Users\\ash\\Desktop\\NLP-DL\\Organising_Complaints_Using_NLP\\Output\\"
# plt.savefig(path + i + ".png", format="png")
```

    Producing Word Cloud 
    


![png](MMT_Tweets_Analysis_files/MMT_Tweets_Analysis_12_1.png)


### Analyze the Sentiments. As we dont have any tagged data we will use the NLTK sentiment polarity Score

### All tweets which have score less than 0 are tags as Negative otherwise Neutral 

*** By looking at tweets i realized that positive does not make sense , as people has sense of complain and request rather than writing anything good.Other score greater than 0 are treated as 2


```python
### Get the sentiments by using NLTK library
sentiments = SentimentIntensityAnalyzer()
### Get the tweets sentiments score
tweets_df['sentiment_compound_polarity']= tweets_df['tweets_text_cleaned'].apply(lambda x:sentiments.polarity_scores(x)['compound'])
### Map the tweets into categories if < 0 negative, if > 0 positive else neutral
tweets_df['Tweets_Sentiment'] = np.where(tweets_df['sentiment_compound_polarity'] < 0,"Negative","Neutral ")
### Look at the distribution of claims
tweets_df['Tweets_Sentiment'].value_counts(normalize = True).plot(kind='bar',title = 'Distribution of Claims by Sentiment')
plt.ylabel('%age Tweets [0-1]')
plt.xlabel('Sentiments')
print (tweets_df['Tweets_Sentiment'].value_counts(normalize = True))
```

    C:\Users\ash\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      after removing the cwd from sys.path.
    C:\Users\ash\Anaconda3\lib\site-packages\ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    

    Neutral     0.616506
    Negative    0.383494
    Name: Tweets_Sentiment, dtype: float64
    


![png](MMT_Tweets_Analysis_files/MMT_Tweets_Analysis_14_2.png)


### Our Hypothesis was correct 40% of tweets are negative in last few days


```python
### Lets look at the word cloud again
```


```python
sentiment_list = tweets_df['Tweets_Sentiment'].unique()
for i in sentiment_list:
    text = " ".join(review.lower() for review in tweets_df[tweets_df['Tweets_Sentiment'] == i]['tweets_text_cleaned'])
    stopwords = set(STOPWORDS)
    stopwords.update(['makemytrip','mmt','please','will'])

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

    # Display the generated image:
    # the matplotlib way:
    print ("Producing Word Cloud for :", i)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show() 
    path = "C:\\Users\\ash\\Desktop\\NLP-DL\\Organising_Complaints_Using_NLP\\Output\\"
    plt.savefig(path + i + ".png", format="png")
```

    Producing Word Cloud for : Neutral 
    


![png](MMT_Tweets_Analysis_files/MMT_Tweets_Analysis_17_1.png)


    Producing Word Cloud for : Negative
    


![png](MMT_Tweets_Analysis_files/MMT_Tweets_Analysis_17_3.png)



    <Figure size 432x288 with 0 Axes>



```python
### Clean the text for Topic Modelling
nlp = spacy.load('en_core_web_sm')
### Remove the stop words using spacy predefined list 
stop_words = nlp.Defaults.stop_words
#### Create a list of puntuation to be removed
import string
symbols = " ".join(string.punctuation).split(" ") 
### As we are doing topic modelling itsa good idea to do lemmatisation - as it uses morphologial analysis

import re

#### Lets define the cleaning function and see how it works
def cleanup_text(docs,logging = False):
    texts = []
    counter = 1
    for doc in docs:
        
        if counter % 5000 == 0 :
            print ("Processed %d of out of %d documents"% (counter,len(docs)))
        counter += 1
        
        doc = nlp(doc) ### We are disabling parser as will nt be using it
        
        
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != "-PRON-"]
        
        tokens =[tok for tok in tokens if tok not in symbols]
        tokens = [tok for tok in tokens if tok not in stop_words]
        tokens = [re.sub('[0-9]', '', i) for i in tokens]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return (pd.Series(texts))
```


```python
tweets_df['tweets_preprocessed'] = cleanup_text(tweets_df['tweets_text_cleaned'])
```

    C:\Users\ash\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    


```python
### Drop the NA columns
tweets_df = tweets_df.dropna().reset_index()
```


```python
### Lets Create the piprle line for NMF models 
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF 
#####  Let extract from act the features from the dataset

print ("Extracting the tf-idf features form NMF")
tfidf_vectorizer = TfidfVectorizer(max_df = 0.5, min_df = 5, max_features = 500, ngram_range = (1,4))

t0 = time()
tfidf = tfidf_vectorizer.fit_transform(tweets_df['tweets_preprocessed'])
print ("done in %0.3fs." % (time() - t0))
```

    Extracting the tf-idf features form NMF
    done in 0.383s.
    


```python
### Latent Drichtlet Aloocations
from sklearn.decomposition import LatentDirichletAllocation
# Fit the NMF model
#### Now we will fit model for 10 diff values of clusters
n_comp = [10]

for comps in n_comp:
    loss1 = []
    t0 = time()
    lda = LatentDirichletAllocation(n_components=comps, max_iter=2,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    lda.fit(tfidf)
    print("done in %0.3fs." % (time() - t0))
```

    done in 1.067s.
    


```python
## Extract The top keywords for the topic
topic_name = {}
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        list1 = "_".join([feature_names[i]
                             for i in topic.argsort()[:-6 - 1:-1]])
        topic_name[topic_idx] = list1
        print(message)
        print (list1)
    return (topic_name)
a= print_top_words(lda, tfidf_vectorizer.get_feature_names(), 10)
```

    Topic #0: lose lockdown current register booking customer provide detail st try
    lose_lockdown_current_register_booking_customer
    Topic #1: credit shell credit shell card use amp rs payment refund original
    credit_shell_credit shell_card_use_amp
    Topic #2: wish assist hey pass makemytrip dgca help refund rt goi
    wish_assist_hey_pass_makemytrip_dgca
    Topic #3: goi wait look hard earn refund month hard earn money earn money
    goi_wait_look_hard_earn_refund
    Topic #4: refund customer reply response service mmt booking respond bad guy
    refund_customer_reply_response_service_mmt
    Topic #5: dm share detail yes offer help long free refer know
    dm_share_detail_yes_offer_help
    Topic #6: day resolve query india issue resolve issue want brand guy buy
    day_resolve_query_india_issue_resolve issue
    Topic #7: thank update status refund official india policy goi official hello update refund
    thank_update_status_refund_official_india
    Topic #8: makemytrip rt big fraud goibibo india kind amp airways month
    makemytrip_rt_big_fraud_goibibo_india
    Topic #9: refund flight cancel book rt ticket money mmt airline receive
    refund_flight_cancel_book_rt_ticket
    


```python
### Assign topic to keywords 
topic_values = lda.transform(tfidf)
tweets_df['topic_assigned'] = topic_values.argmax(axis = 1)
tweets_df['topic_name '] = tweets_df['topic_assigned'].map(a)
```


```python
### Plot the distribution by topic
plt.figure(figsize=(20,15))
plt.rcParams.update({'font.size': 22})
tweets_df['topic_name '].value_counts().plot(kind = 'barh',title = 'Number of Tweets By Topics')
plt.xlabel("Number of Tweets")
plt.ylabel("Topic - top 6 keywords")
```




    Text(0, 0.5, 'Topic - top 6 keywords')




![png](MMT_Tweets_Analysis_files/MMT_Tweets_Analysis_25_1.png)



```python

```
