"""\
This file performs sentiment analysis.

Usage: sentiment_analysis.py path_to_dataframe
"""

import pandas as pd
import numpy as np

import re
import argparse
from tqdm import tqdm

from transformers import pipeline

config = {
    'sentiment_model' : 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'search_strings' : ['semaglutide', 'rybelsus', 'wegovy', 'ozempic',
                    'retatrutide',
                    'dulaglutide', 'trulicity',
                    'tirzepatide', 'mounjaro',
                    'liraglutide', 'saxenda',
                    'exenatide', 'bydureon', 'byetta',
                    'lixisenatide', 'adlyxin']
}

class SentimentAnalysis():
    """
    Class that handles sentiment analysis.
    """

    def __init__(self):
        """"
        Initiate the class w/ the appropriate sentiment model.
        """

        self.config = config
        self.sentiment_model = pipeline('sentiment_analysis', model=self.config['sentiment_model'], tokenizer=self.config['sentiment_model'], return_all_scores=True)

    def load_data_frame(self):
        """
        Load our Reddit post/comments dataframe.
        """

        df = pd.read_excel(self.config['data'])
        self.df = self.preprocess_dataframe(df)
        self.texts = self.df['content'].to_list()

    @staticmethod
    def preprocess_dataframe(df):
        """
        Preprocess dataframe for topic modeling, if this has not already been performed.
        """
        # Fill empty cells and remove some weird html tags
        df['body'].fillna("", inplace=True)
        df['body'] = df['body'].str.replace("http\S+", "")
        df['body'] = df['body'].str.replace("\\n", " ")
        df['body'] = df['body'].str.replace("&gt;", "")
        
        # Get rid of extra spaces
        df['body'] = df['body'].str.replace('\s+', ' ', regex=True)
        
        # Remove those too small.
        df['body_len'] = df['body'].str.len()
        df = df.query('body_len >= 25')

    def perform_sentiment_analysis(self):
        """
        Perform sentiment analysis.
        """

        # Create output df.
        sentiments = pd.DataFrame(columns=['negative', 'neutral', 'positive'])

        # Calculate sentiment
        for i, text in tqdm(enumerate(self.texts)):
            sentiments.loc[i, :] = self.find_string_sentiment(self.sentiment_model, text)

        # Map sentiment to a singular value based on highest probability assignment.
        sentiments['net'] = np.argmax(np.array(sentiments.loc[:, ['negative', 'neutral', 'positive']]), axis=1)

        # Merge this into the main dataframe.
        sentiments.index = self.df.index
        self.sentiments = self.df.join(sentiments)

        # Save
        self.sentiments.to_excel(self.config['output'])

    def find_string_sentiment(self, text):
        
        len_text = len(text)
        indices = [match.start() for match in re.finditer('|'.join(self.config['search_strings']), text.lower())]
        sents = np.empty((len(indices), 3))
        
        for i, index in enumerate(indices):

            if (index - 256) < 0:
                start_idx = 0
            else:
                start_idx = (index - 256)

            if (index + 256) > len_text:
                end_idx = len_text
            else:
                end_idx = (index + 256)
        
            sentiment_i = self.sentiment_model(text[start_idx:end_idx])
            sents[i, :] = sentiment_i[0]['score'], sentiment_i[1]['score'], sentiment_i[2]['score']
        
        return np.mean(sents, axis=0)
    
if __name__ == '__main__':
    
    # Enter input data via argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type='str', help='Path to dataset')
    parser.add_argument('output', type='str', help='Path to save sentiment analysis results.')
    args = parser.parse_args()

    config['data'] = args.data
    config['output'] = args.output

    sentiment_analysis = SentimentAnalysis(config)
    sentiment_analysis.perform_sentiment_analysis()