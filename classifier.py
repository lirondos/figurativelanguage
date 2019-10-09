from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from operator import itemgetter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import numpy as np
import os, re
from nltk.corpus import stopwords
import annotation


STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('<[^>]+>', '', text)  # erase HTML tags
    text = re.sub(r'[^A-Za-z]+', ' ', text)
    return text.lower()



def stem_tokens(stemmer, words):
    # text = re.sub(r'\d+', '', text)
    words_without_stopwords = []
    for word in words:
        if word not in STOPWORDS:
            # print(word)
            words_without_stopwords.append(word)
    return [stemmer.stem(token) for token in words_without_stopwords]


# In[206]:


def bigrams(words):
    bigrams = list(nltk.bigrams(words))
    bigrams_lower = [(bigram1.lower(), bigram2.lower()) for (bigram1, bigram2) in bigrams]
    bigram_not_stopwords = []
    for bigram in bigrams_lower:
        # remove stopwords bigrams
        if bigram[0] not in STOPWORDS or bigram[1] not in STOPWORDS:
            bigram_not_stopwords.append(bigram)
    return bigram_not_stopwords
    # return word_tokenize(text)


# In[207]:


def trigrams(words):
    trigrams = list(nltk.trigrams(words))
    trigrams_lower = [(trigram1.lower(), trigram2.lower(), trigram3.lower()) for (trigram1, trigram2, trigram3) in
                      trigrams]
    trigram_not_stopwords = []
    for trigram in trigrams_lower:
        # remove stopwords bigrams
        if trigram[0] not in STOPWORDS and trigram[2] not in STOPWORDS:
            trigram_not_stopwords.append(trigram)
    return trigram_not_stopwords
    # return word_tokenize(text)


def extract_words(path_to_file, stemmer):
    with open(path_to_file, 'r') as file:
        list_words = file.read().splitlines()
        list_words_stemmed = [stemmer.stem(token) for token in list_words]
    # print(list_words)
    return list_words_stemmed


def extract_feat_vocab(file):
    data_frame = pd.read_csv(file, encoding='latin1')
    feat_vocab = dict()
    for index, row in data_frame[data_frame['type'] == 'train'].iterrows():
        text = clean_text(row['tweet'])
        tokens = word_tokenize(text)
        for token in tokens:
            feat_vocab[token] = feat_vocab.get(token, 0) + 1
    return feat_vocab


def select_features(feat_vocab, most_freq=100, least_freq=5000):
    sorted_feat_vocab = sorted(feat_vocab.items(), key=itemgetter(1), reverse=True)
    feat_dict = dict(sorted_feat_vocab[most_freq:len(sorted_feat_vocab) - least_freq])
    return set(feat_dict.keys())
    """
    feat_dict = dict()
    for key, value in feat_vocab.items():
        if value>1:
            feat_dict[key] = value
    return set(feat_dict.keys())
    """

def featurize(csv_file, feat_vocab):
    cols = ['_type_', '_label_']
    cols.extend(list(feat_vocab))
    #cols.extend(['positive_words', 'negative_words', 'subjectivity_words'])
    data_frame = pd.read_csv(csv_file, encoding='latin1', engine='python')
    #     dtype={'FULL': 'str', 'COUNT': 'int'}
    row_count = data_frame.shape[0]
    feat_data_frame = pd.DataFrame(index=np.arange(row_count), columns=cols)
    feat_data_frame.fillna(0, inplace=True)  # inplace: mutable
    for index, row in data_frame.iterrows():
        feat_data_frame.loc[index, '_type_'] = row['type']
        feat_data_frame.loc[index, '_label_'] = row['label']
        text = clean_text(row['tweet'])
        tokens = word_tokenize(text)
        for token in tokens:
            if token.lower() in feat_vocab:
                feat_data_frame.loc[index, token] += 1
    return feat_data_frame


def vectorize(feature_csv, split="train"):
    df = pd.read_csv(feature_csv, encoding='latin1', engine='python')
    df = df[df['_type_'] == split]
    df.fillna(0, inplace=True)
    data = list()
    for index, row in df.iterrows():
        datum = dict()
        datum['bias'] = 1
        for col in df.columns:
            if not (col == "_type_" or col == "_label_" or col == 'index'):
                datum[col] = row[col]
        data.append(datum)
    #print(data)
    vec = DictVectorizer()
    data = vec.fit_transform(data).toarray()
    #print(data.shape)
    labels = df._label_.as_matrix()
    #print(labels.shape)
    return data, labels



def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    print("Shape of model coefficients and intercepts: {} {}".format(model.coef_.shape, model.intercept_.shape))
    return model


def test_model(X_test, y_test, model):
    predictions = model.predict(X_test)
    report = classification_report(predictions, y_test)
    accuracy = accuracy_score(predictions, y_test)
    return accuracy, report


def classify(feat_csv):
    X_train, y_train = vectorize(feat_csv)
    X_test, y_test = vectorize(feat_csv, split='test')
    model = LogisticRegression(multi_class='multinomial', penalty='l2', solver='lbfgs', max_iter=20,
                               verbose=1)
    model = train_model(X_train, y_train, model)
    accuracy, report = test_model(X_test, y_test, model)
    print(report)


def select_rows(from_csv, to_csvfile):
    from_df = pd.read_csv(from_csv, encoding='latin1', engine='python')
    to_df = pd.DataFrame(columns=['type', 'tweet', 'label'])
    to_index = 0
    for index, row in from_df.iterrows():
        if index % 1 == 0:  # 10 original
            to_df.loc[to_index] = [row['type'], row['review'], row['label']]
            to_index += 1
    to_df.to_csv(to_csvfile, encoding='latin1')


# In[217]:


if __name__ == '__main__':
    #replace filename to run on different data set
    file = "sarcasm vs everything.csv"
    feat_vocab = extract_feat_vocab(file)
    selected_feat_vocab = select_features(feat_vocab, 1, 50)
    feat_data_frame = featurize(file, selected_feat_vocab)
    featfile = os.path.join(os.path.curdir, "features.csv")
    feat_data_frame.to_csv(featfile, encoding='latin1', index=False)
    classify('features.csv')

