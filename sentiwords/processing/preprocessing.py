from nltk import TweetTokenizer
import pandas as pd
import re
from rfc3987 import get_compiled_pattern

class Preprocessor():

    def __init__(self, reduce_len=True, preserve_case=False, strip_handles=True, strip_urls=True, stopwords=[]):
        self.tokenizer = TweetTokenizer(reduce_len=reduce_len, strip_handles=strip_handles, preserve_case=preserve_case)
        self.strip_urls = strip_urls
        self.stopwords = stopwords
        self.url_token = '<URL>'
        self.url_pattern = re.compile(r"""((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,‌​3}[.]|[a-z0-9.\-]+[.‌​][a-z]{2,4}/)(?:[^\s‌​()<>]+|(([^\s()<‌​>]+|(([^\s()<>]+‌​)))*))+(?:&#‌​40;([^\s()<>]+|((‌​;[^\s()<>]+)))*&‌​#41;|[^\s`!()[&#‌​93;{};:'".,<>?«»“”‘’‌​]))""", re.DOTALL)

    def tokenize_tweet(self, tweet):
        if self.strip_urls:
            #tweet = self.url_pattern.sub(self.url_token, tweet)
            tweet = self.url_pattern.sub(self.url_token, tweet)
        tokens = self.tokenizer.tokenize(tweet)
        tokens = filter(lambda x: x not in self.stopwords, tokens)
        tokens = list(map(lambda x: x.replace(' ', ''), tokens))
        return tokens

    def preprocess_csv(self, csv, columns=['sentiment', 'id', 'date', 'status', 'user', 'text']):
        df = pd.read_csv(csv, sep=',', header=None, names=columns)
        df['tokens'] = df['text'].apply(self.tokenize_tweet)
        return df['tokens'].values
# if __name__=='__main__':
#     processor = Preprocessor()
#     processor.preprocess_csv('/home/maurice/Downloads/training.1600000.processed.noemoticon.csv')
