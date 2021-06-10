#!/usr/bin/env python
# coding: utf-8

# # Programming for Sentiment Analysis for Tweets

# ## 1. Loading Libraries

# In[1]:


##Pandas and Numpy
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)

from numpy import array
from numpy import asarray
from numpy import zeros
import seaborn as sns

#Pandas Profiling numpy
from pandas_profiling import ProfileReport

import re #For removing numbers
import string #For removing punchuations

#For removing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))

#For Lemmatization
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

#KMeans Clustering
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler

# Plot the word cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.express as px
import plotly.figure_factory as ff
from palettable.colorbrewer.qualitative import Pastel1_7

from collections import defaultdict
from nltk.corpus import stopwords

#Sentiment Analysis technique with tf-idf 
from sklearn.feature_extraction.text import TfidfVectorizer

## NaiveBayes Classifier
from sklearn.naive_bayes import MultinomialNB

#Splitting Dataset
from sklearn.model_selection import train_test_split

#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics

#LSTIM and Pre-trained Word Embeddings
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout, Flatten, GlobalMaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub

import tensorflow_hub as hub
from sklearn import tree
import pydotplus  # pip install pydotplus
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# ## 2. Importing Dataset

# In[2]:


tweetDataset = pd.read_csv('DataSet/tweet_sentiment_dataset.csv')  


# ### 2.1. Overviewing Dataset

# In[3]:


print(f"Dataset has {tweetDataset.shape[0]} rows, and {tweetDataset.shape[1]} columns!")


# In[4]:


# Generate 5 random rows from the dataset
tweetDataset.sample(5) 


# In[5]:


#Check to see if there are any NULL values in the dataset
tweetDataset.isnull().sum()


# In[6]:


print(round(tweetDataset[tweetDataset['label (depression result)'] == 0].shape[0]/tweetDataset.shape[0] *100, 2), "% of the data is of label 0 ")
print(round(tweetDataset[tweetDataset['label (depression result)'] == 1].shape[0]/tweetDataset.shape[0] *100, 2), "% of the data is of label 1 ")


# ### 2.2. Statistical Viewing of text

# In[7]:


# Plotting Pie Chart 
plt.pie(tweetDataset['label (depression result)'].value_counts(),
        shadow=True,
        explode=[0.1,0.1],
        labels=["Not Having Depression ","Having Depression"],
        colors=["green","red"], #specify custom colors
        autopct='%.3f%%', #format for labels of wedges (numerical value)
        radius=1.1, #radius of the the pie
        textprops={"fontsize":10} #set text properties
       )
plt.title("Depression Statistics")


# In[8]:


# Plotting BarGraph with Title, Lables, Legend
plt.style.use('ggplot')
plt.figure(figsize = (10, 5))
(tweetDataset['label (depression result)'].value_counts()).plot(kind = 'bar', color='#77AC30',width=.5)
plt.title("Depression Statistics",weight='bold')
plt.ylabel("No. of Person",weight='bold')
plt.xlabel("Depression Status",weight='bold')
plt.xticks(rotation=0)
plt.text(-0.20,164,"Not Having Depression")
plt.text(0.85,85,"Having Depression")
plt.show()


# In[9]:


tweetDataset.groupby("label (depression result)").sample(n=5, random_state=1)


# From the 'message to examine' column, it can be observed that our textual data contains all of alphabetical, numerical, hyperlinks, random junk characters, as well as other sets of special character values.

# In[10]:


tweetDataset.loc[8047]['message to examine']


# In[11]:


charLength = tweetDataset['message to examine'].str.len()
charLength.hist()


# Most of the text characters in 'message to examine' column are of length 50, and a few of them have as many as 370.

# In[12]:


# Count the number of sentences in each lists!
tweetDataset['countSentences']=tweetDataset["message to examine"].apply(lambda x: len(re.findall("\n",str(x)))+1)
# Count the number of words in each lists!
tweetDataset['countWords']=tweetDataset["message to examine"].apply(lambda x: len(str(x).split()))


# In[13]:


# There aren't much sentences, if not any. Hence the lengh of sentences have been kept to bare minimum
tweetDataset['countSentences'].loc[tweetDataset['countSentences']>0] = 0
plt.figure(figsize=(12,6))
## sentences
plt.subplot(121)
plt.suptitle("Are longer sentences/words more depressing?",fontsize=20)
sns.violinplot(y='countSentences',x='label (depression result)', data=tweetDataset,split=True)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('# of sentences', fontsize=12)
plt.title("Number of sentences in each category", fontsize=15)
# words
tweetDataset['countWords'].loc[tweetDataset['countWords']>50] = 50
plt.subplot(122)
sns.violinplot(y='countWords',x='label (depression result)', data=tweetDataset,split=True,inner="quart")
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title("Number of words in each category", fontsize=15)

plt.show()


# ## 3. Dataset Analysis using Pandas Profiling

# In[14]:


profile = ProfileReport(tweetDataset, title='Pandas Profiling Report', explorative=True)
profile.to_notebook_iframe()


# ## 4. Text Preprocessing - Phase I
#    - Removing Numbers
#    - Convering all the tweets into Lowercase
#    - Removing Weblinks
#    - Removing Twitter Mentions
#    - Romoving Punctuations
#    - Removing Stopwords
#    - Finding Frequently Used Words
#        - Words Count Visualization on Bar Garph
#        - Words Count Visualization on Tree Map
#        - Words Count Visualization on DoNut Plot
#    - Finding Rarely Used Words
#    - Lemmatization
#    - Replacing Short Words

# In[15]:


# regex to remove all Non-Alpha Numeric and space
special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)
# regex to replace all numeric
replace_numbers=re.compile(r'\d+',re.IGNORECASE)
#regex for String Punctuations
PUNCT_TO_REMOVE = string.punctuation
#Set STOPWORDS
STOPWORDS = set(stopwords.words('english'))

def textPreprocessing(text, stem_words=False):
    text = text.lower() #convert to lowercase
    
    #replace short words
    text = re.sub("ain't", "am not", text)
    text = re.sub("aren't", "are not", text)
    text = re.sub("can't", "cannot", text)
    text = re.sub("'cause", "because", text)
    text = re.sub("could've", "could have", text)
    text = re.sub("couldn't", "could not", text)
    text = re.sub("didn't", "did not", text)
    text = re.sub("doesn't", "does not", text)
    text = re.sub("don't", "do not", text)
    text = re.sub("hadn't", "had not", text)
    text = re.sub("hasn't", "has not", text)
    text = re.sub("haven't", "have not", text)
    text = re.sub("i'm", "I am", text)
    text = re.sub("'em", "them", text)
    text = re.sub("i've", "I have", text)
    text = re.sub("isn't", "is not", text)
    text = re.sub("let's", "let us", text)
    text = re.sub("they're", "they are", text)
    text = re.sub("they've", "they have", text)
    text = re.sub("wasn't", "was not", text)
    text = re.sub("we'll", "we will", text)
    text = re.sub("we're", "we are", text)
    text = re.sub("weren't", "were not", text)
    text = re.sub("you're", "you are", text)
    text = re.sub("you've", "you have", text)
    
    text = re.sub(r"http\S+", "", text) #replace all weblinks with empty string
    text = re.sub('@[\w]+','',text) #replace all twitter mentions with empty strings
    text = replace_numbers.sub('', text) #replace numbers
    text = special_character_removal.sub('',text) #replace special characters
    text = text.translate(str.maketrans('','', PUNCT_TO_REMOVE)) #remove punctuations
    words = [word for word in text.split() if not word in STOPWORDS] #remove stopwords
    cleanText = " ".join(words)
    return cleanText


# In[16]:


tweetDataset['clean_tweets'] = tweetDataset['message to examine'].apply(lambda x: textPreprocessing(x))


# In[17]:


tweetDataset.loc[4205]['clean_tweets']


# In[18]:


#Finding most frequently used words
from collections import Counter
cnt = Counter()

for text in tweetDataset['clean_tweets'].values:
    for word in text.split():
        cnt[word] += 1
        
topTenCommonWord =  pd.DataFrame(cnt.most_common(10))       
topTenCommonWord.columns = ['Common_words','count']
topTenCommonWord.style.background_gradient(cmap='Blues')


# Since, this is a sentiment analysis, depression word is important for sentiment analysis

# In[19]:


# Visualizing frequently used words on Bar Garph
fig = px.bar(topTenCommonWord, x="count", y="Common_words", title='Freuently Used Words in Tweets', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()


# In[20]:


# Visualizing frequently used words on Tree Map
fig = px.treemap(topTenCommonWord, path=['Common_words'], values='count',title='Tree of Freuently Used Words in Tweets')
fig.show()


# In[21]:


#Visualizing frequently used words on DoNut Plot
plt.figure(figsize=(16,10))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.rcParams['text.color'] = 'black'
plt.pie(topTenCommonWord['count'], labels=topTenCommonWord.Common_words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Most Frequently Words in Tweets')
plt.show()


# In[22]:


#Finding rarely used words
n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
print(RAREWORDS)


# In[23]:


#Removing rarely used words
def removeRareWords(text):
    words = [word for word in text.split() if not word in RAREWORDS] #remove stopwords
    cleanText = " ".join(words)
    return cleanText

tweetDataset['cleanTweetsComplete'] = tweetDataset['clean_tweets'].apply(lambda x: removeRareWords(x))


# In[24]:


tweetDataset.loc[8047]['cleanTweetsComplete']


# #### Lemmatization

# In[25]:


lemmatizer = WordNetLemmatizer()
wordnet_map = {"n": wordnet.NOUN, "v": wordnet.VERB, "j": wordnet.ADJ, "r": wordnet.ADV}

def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.VERB)) for word, pos in pos_tagged_text])

tweetDataset["textLemma"] = tweetDataset['cleanTweetsComplete'].apply(lambda text: lemmatize_words(text))


# Now our text is clean and ready for training. We have used many columns for cleaning the data. Let's delete all the unwanted columns.

# In[26]:


finalTweetDataset = tweetDataset[['Index','label (depression result)','cleanTweetsComplete']]
finalTweetDataset.columns = ['Index','Labels','Tweets']
finalTweetDataset.sample(5)


# ## 5. Word Cloud Plot

# In[27]:


sentences = finalTweetDataset['Tweets'].tolist()
joined_sentences = " ".join(sentences)

plt.figure(figsize = (12,8))
plt.imshow(WordCloud().generate(joined_sentences));


# ### 5.2. Visualizing postive and negative tweets

# #### 5.2.1. Visualizing postive tweets

# In[28]:


positiveTweets = finalTweetDataset[finalTweetDataset['Labels'] == 0]
positiveSentences = positiveTweets['Tweets'].tolist()
positiveStrings = " ".join(positiveSentences)

plt.figure(figsize = (12,8))
plt.imshow(WordCloud().generate(positiveStrings));


# In[29]:


#counter
cntPositive = Counter()

for text in positiveTweets['Tweets'].values:
  for word in text.split():
    cntPositive[word] += 1

cntPositiveWords = pd.DataFrame(cntPositive.most_common(10))
cntPositiveWords.columns = ['positive_words','count']
cntPositiveWords.style.background_gradient(cmap='Greens')


# In[30]:


#bar graph for Postive Sentiment Tweets
fig = px.bar(cntPositiveWords, x="count", y="positive_words", title='Postive Sentiment Tweets', orientation='h', 
             width=700, height=700,color='positive_words')
fig.show()


# In[31]:


#tree map for Postive Sentiment Tweets
fig = px.treemap(cntPositiveWords, path=['positive_words'], values='count',title='Tree of Postive Sentiment Tweets')
fig.show()


# In[32]:


#doNut Plot for Postive Sentiment Tweets
plt.figure(figsize=(16,10))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.rcParams['text.color'] = 'black'
plt.pie(cntPositiveWords['count'], labels=cntPositiveWords.positive_words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Postive Sentiment Tweets')
plt.show()


# #### 5.2.2. Visualizing negative tweets

# In[33]:


negativeTweets = finalTweetDataset[finalTweetDataset['Labels'] == 1]
negativeSentences = negativeTweets['Tweets'].tolist()
negativeStrings = " ".join(negativeSentences)

plt.figure(figsize = (12,8))
plt.imshow(WordCloud().generate(negativeStrings));


# In[34]:


#counter
cntNegativeWord = Counter()

for text in negativeTweets['Tweets'].values:
  for word in text.split():
    cntNegativeWord[word] += 1

cntNegativeWord = pd.DataFrame(cntNegativeWord.most_common(10))
cntNegativeWord.columns = ['negative_words','count']
cntNegativeWord.style.background_gradient(cmap='Reds')


# In[35]:


#bar graph for Negative Sentiment Tweets
fig = px.bar(cntNegativeWord, x="count", y="negative_words", title='Negative Sentiment Tweets', orientation='h', 
             width=700, height=700,color='negative_words')
fig.show()


# In[36]:


#tree map for Negative Sentiment Tweets
fig = px.treemap(cntNegativeWord, path=['negative_words'], values='count',title='Tree of Negative Sentiment Tweets')
fig.show()


# In[37]:


#doNut Plot for Negative Sentiment Tweets
plt.figure(figsize=(13,8))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.rcParams['text.color'] = 'black'
plt.pie(cntNegativeWord['count'], labels=cntNegativeWord.negative_words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Negative Sentiment Tweets')
plt.show()


# ## 6. Text Preprocessing - Phase II
#    - **TF-IDF**
#    - **word2vec**

# ### 6.1. Basic Sentiment Analaysis
# First, we will build our model with Basic Sentiment Analysis technique with **TF-IDF** and **NaiveBayes Classifier**

# In[38]:


cv = TfidfVectorizer()
tfidf = cv.fit_transform(finalTweetDataset['Tweets'])

#Splitting the dataset into 80-20 ratio between train and test sets
tfX_train, tfX_test, tfy_train, tfy_test = train_test_split(tfidf, finalTweetDataset['Labels'], test_size = 0.2)


# In[39]:


print(f"Train Data Shape {tfX_train.shape},Train Labels Shape {tfy_train.shape}")
print(f"Test Data Shape {tfX_test.shape},Test Labels Shape {tfy_test.shape}")


# #### 6.1.2. Models and Evaluation

# In[40]:


mnb = MultinomialNB()
mnb.fit(tfX_train, tfy_train)

y_pred_mnb = mnb.predict(tfX_test)
print(f'Accuracy score is : {round(accuracy_score(tfy_test, y_pred_mnb), 2)}')


# In[41]:


cf = confusion_matrix(tfy_test, y_pred_mnb, labels = [1,0])

x_axis_labels = ["Positive(1)","Negative(0)"]
y_axis_labels = ["Positive(1)","Negative(0)"]

plt.figure(figsize = (8,6))
sns.set(font_scale=1)
sns.heatmap(cf, xticklabels = x_axis_labels, yticklabels = y_axis_labels, annot = True, fmt='g',annot_kws = {'size': 16})
plt.xlabel("Actual Class", fontsize = 20)
plt.ylabel("Predicted Class", fontsize = 20)
plt.show()


# ### 6.2. Advance Sentiment Analaysis
# We will improve our model with **LSTM** and **Pre-trained Word Embeddings**
# 
# The use of pre-trained word embeddings.
# 
# The theory is that these pre-trained vectors already have words with similar semantic meaning close together in vector space, e.g. "sad", "depressed", "bad" are nearby. This gives our embedding layer a good initialization as it does not have to learn these relations from scratch.

# ###### Get max token counts from train data, so we use this number as fixed length input to RNN cell

# In[42]:


# Loading pre-trained word embeddings for word2vec
embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")

def get_max_length(df):
    max_length = 0
    for row in finalTweetDataset['Tweets']:
        if len(row.split(" ")) > max_length:
            max_length = len(row.split(" "))
    return max_length

get_max_length(finalTweetDataset['Tweets'])


# ###### Get word2vec value for each word in sentence concatenate word in numpy array, so we can use it as RNN input

# In[43]:


def get_word2vec_enc(tweets):
    encoded_tweets = []
    for tweet in tweets:
        tokens = tweet.split(" ")
        word2vec_embedding = embed(tokens)
        encoded_tweets.append(word2vec_embedding)
    return encoded_tweets


# ###### For short sentences, we prepend zero padding so all input to RNN, has same length

# In[44]:


def get_padded_encoded_tweets(encoded_tweets):
    padded_tweets_encoding = []
    for enc_tweet in encoded_tweets:
        zero_padding_cnt = max_length - enc_tweet.shape[0]
        pad = np.zeros((1, 250))
        for i in range(zero_padding_cnt):
            enc_tweet = np.concatenate((pad, enc_tweet), axis = 0)
        padded_tweets_encoding.append(enc_tweet)
    return padded_tweets_encoding

def sentiment_encode(sentiment):
    if sentiment == 0:
        return [0,1]
    else:
        return [1,0]


# ###### Encode text value to numeric value and sentiment

# In[45]:


def preprocess(finalTweetDataset):
  # Encoding text value to numeric value
    tweets = finalTweetDataset['Tweets'].tolist()
    encoded_tweets = get_word2vec_enc(tweets)
    
    padded_encoded_tweets = get_padded_encoded_tweets(encoded_tweets)
    
    # Encoding sentiment
    sentiments = finalTweetDataset['Labels'].tolist()
    encoded_sentiment = [sentiment_encode(sentiment) for sentiment in sentiments]
    
    X = np.array(padded_encoded_tweets)
    Y = np.array(encoded_sentiment)

    return X, Y


# ###### Preprocess 

# In[46]:


max_length = get_max_length(finalTweetDataset)

tdf = finalTweetDataset.sample(frac = 1)
train = tdf[:8000]
test = tdf[8000:]


# In[47]:


print(f"Train Shape {train.shape},Test Shape {test.shape}")

train_X, train_Y = preprocess(train)
test_X, test_Y = preprocess(test)


# #### 6.2.1. Building a Model using LSTM

# In[48]:


# Building a LTSM model, and training
model = Sequential()
model.add(LSTM(32))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit(train_X, train_Y, epochs = 10)

model.summary()


# #### 6.2.2. Testing the dataset

# In[49]:


score, acc = model.evaluate(test_X, test_Y, verbose = 0)
print(f"Test Score: {score}, Test Accuract: {acc}")


# #### 6.2.3. Confusion Matrix

# In[50]:


y_pred = model.predict(test_X)

matrix = metrics.confusion_matrix(test_Y.argmax(axis = 1), y_pred.argmax(axis = 1), labels = [1,0])

x_axis_labels = ["Positive(1)","Negative(0)"]
y_axis_labels = ["Positive(1)","Negative(0)"]

plt.figure(figsize = (8,6))
sns.set(font_scale=1)
sns.heatmap(matrix, xticklabels = x_axis_labels, yticklabels = y_axis_labels, annot = True, fmt='g',annot_kws = {'size': 16})
plt.xlabel("Actual Class", fontsize = 20)
plt.ylabel("Predicted Class", fontsize = 20)
plt.show()


# ## 7. Transfer Learning with an Embedding Layer

# In[51]:


X_train, X_test, y_train, y_test = train_test_split(finalTweetDataset['Tweets'], finalTweetDataset['Labels'], test_size = 0.2)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# In[52]:


vocab_size = len(tokenizer.word_index) + 1

maxlen = 50

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# In[53]:


embeddings_dictionary = dict()
glove_file = open('./glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()


# In[54]:


embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# In[55]:


model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()


# In[56]:


history = model.fit(X_train, y_train, epochs = 5, batch_size = 64, verbose = 1, validation_split = 0.1)


# In[57]:


score, acc = model.evaluate(X_test, y_test, verbose = 0)
print(f"Test Score: {score}, Test Accuract: {acc}")


# In[58]:


y_pred_model = model.predict(X_test)
cf = confusion_matrix(y_test, y_pred_mnb, labels = [1,0])

x_axis_labels = ["Positive(1)","Negative(0)"]
y_axis_labels = ["Positive(1)","Negative(0)"]

plt.figure(figsize = (8,6))
sns.set(font_scale=1)
sns.heatmap(cf, xticklabels = x_axis_labels, yticklabels = y_axis_labels, annot = True, fmt='g',annot_kws = {'size': 16})
plt.xlabel("Actual Class", fontsize = 20)
plt.ylabel("Predicted Class", fontsize = 20)
plt.show()


# In[59]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

