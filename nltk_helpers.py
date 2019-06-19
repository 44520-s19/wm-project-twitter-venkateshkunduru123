# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:30:45 2019

@author: S534845
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_sentiments(text):
    nltk.download('vader_lexicon')
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def split_sentiments(sentiments):
    xs = [sent['neg'] for sent in sentiments]
    ys = [sent['neu'] for sent in sentiments]
    zs = [sent['pos'] for sent in sentiments]
    return xs,ys,zs