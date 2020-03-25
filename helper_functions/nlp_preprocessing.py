import logging
import re
import nltk
import numpy as np
import string
import unicodedata

from collections import OrderedDict
from nltk import FreqDist, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def strip_multiple_whitespaces(s):
    """
    Remove repeating whitespace characters (spaces, tabs, line breaks) from `s`
    using RE_WHITESPACE
    """
    RE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)
    
    return RE_WHITESPACE.sub(" ", s)


def strip_short(s, minsize=3):
    """
    Remove words with length lesser than `minsize` from `s`.
    """
    return " ".join(e for e in s.split() if len(e) >= minsize)


def strip_punctuation(s):
    """
    Replace punctuation characters with spaces in `s` using RE_PUNCT`.
    """
    RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)
    
    return RE_PUNCT.sub(" ", s)


def strip_tags(s):
    """
    Remove tags from `s` using RE_TAGS`.
    """
    RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)
    
    return RE_TAGS.sub("", s)


def strip_numeric(s):
    """
    Remove digits from `s` using :const:`~gensim.parsing.preprocessing.RE_NUMERIC`.
    """
    RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)
    
    return RE_NUMERIC.sub("", s)


def deaccent(s):
    """
    Remove letter accents from the given string. 
    source: https://github.com/RaRe-Technologies/gensim/blob/ec222e8e3e72608a59805040eadcf5c734a2b96c/gensim/utils.py#L177
    """
    norm = unicodedata.normalize("NFD", s)
    result = ('').join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)


def clean_string_fields(x):
    
    extra_removals = {
    "xxxxxxx", "xxxxx", "xxxxx",
    "•", "∼", "", "β", "abstract", "title"
    }
    
    
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    stopwords_en = set(stopwords.words("english"))
    
    x = (
        x.lower()
        .replace("\r\n\r\n", " ")
    )
    x = strip_multiple_whitespaces(x) 
    x = strip_tags(x)
    x = strip_punctuation(x)
    x = deaccent(x)
    x = strip_short(x)
    x = strip_numeric(x)
    x = word_tokenize(x)
    x = " ".join(word for word in x if word not in stopwords_en|extra_removals)
    x = stemmer.stem(x)
    x = lemmatizer.lemmatize(x)

    return x


# nltp steps
def get_word_frequency_table(series, freq_count):
    """
    get a filtered, sorted word frequency dict from a series containing strings
    Args
     series: pd.Series
     freq_count: threshold above which words will be retained in table
    """
    fdist = FreqDist()
    for k, v in series.items():
        for s in sent_tokenize(v):
            for word in word_tokenize(s):
                fdist[word] += 1

    fdist_filtered = dict((word, freq) for word, freq in fdist.items() if freq >= freq_count)
    
    # sort
    fdist_filtered = {k: v for k, v in sorted(fdist_filtered.items(), 
                                              key = lambda x: x[1], reverse=True)}
    
    return fdist_filtered


def get_top_n_grams(corpus, n_count = None):
    """
    get a sorted list of tuples with word frequencies filtered on n-grams
    Args
     series: pd.Series
     n_count: number of n_grams
    """
    vec = CountVectorizer(ngram_range = (n_count, n_count)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return words_freq
