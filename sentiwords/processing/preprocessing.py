"""Module that handles Tweet pre-processing.
author: Maurice Gercuk
"""
import pandas as pd
import re
from nltk import TweetTokenizer, word_tokenize
from html import unescape

# Found at https://gist.github.com/gruber/8891611
URL_REGEX = re.compile(
    r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|tensorflow embedding lookup defaultgh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
)

USER_PATTERN = re.compile(r'\B(@\w*)+')

# probaly matches too many things, but at least seems to catch all email addresses
EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+')

HTML_TAG_PATTERN = re.compile('<.*?>')

# https://stackoverflow.com/questions/13252101/regular-expressions-match-floating-point-number-but-not-integer top rated answer
NUMBER_PATTERN = re.compile(
    r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?')


class Preprocessor():
    """Class storing configured pre-processing options and offers methods
    to tokenize tweets and preprocess whole csv files accorfingly.
    """

    def __init__(self, reduce_len=True, preserve_case=False, stopwords=[]):
        """Initialize a Preprocessor object.
        arguments:
           reduce_len: Whether repeated occurences of letters in words should
                       be shortened to at most 3 letter. E.g hellooooooo -> hellooo
           preserve_case: Whether the case of words should be preserved.
           stopwords: List of words that should be filtered out of the tokenized
                      tweets.
        """
        self.tokenizer = TweetTokenizer(
            reduce_len=reduce_len, preserve_case=preserve_case)
        self.stopwords = stopwords
        self.url_token = '<url>'
        self.user_token = '<user>'
        self.email_token = '<email>'
        self.tag_token = '<tag>'
        self.number_token = '<number>'

    def tokenize_tweet(self, tweet):
        """Tokenizes a tweet according to the preprocessors parameters.
        arguments:
           tweet: The input tweet (a String)
        """
        tweet = unescape(tweet)
        tokens = self.tokenizer.tokenize(tweet)
        tokens = map(lambda x: x.replace('#', ''), tokens)
        tokens = (self.__maybe_replace_with_token(word) for word in tokens)
        tokens = filter(lambda x: x not in self.stopwords, tokens)
        tokens = map(lambda x: x.replace(' ', ''), tokens)
        return list(tokens)

    def __maybe_replace_with_token(self, word):
        """Maybe replaces a word with a matching token.
        arguments:
           word: The word that might be replaced by a token.
        returns:
           one of the tokens defined in Preprocessor if the word matches the
           corresponding pattern. Otherwise, returns the word."""
        if HTML_TAG_PATTERN.match(word): return self.tag_token
        if NUMBER_PATTERN.match(word): return self.number_token
        if USER_PATTERN.match(word): return self.user_token
        if EMAIL_PATTERN.match(word): return self.email_token
        if URL_REGEX.match(word): return self.url_token
        else: return word

    def preprocess_csv(
            self,
            csv,
            columns=['sentiment', 'id', 'date', 'status', 'user', 'text']):
        """Preprocess a whole csv file with tweets.
        arguments:
           csv: Filepath of input csv file.
           columns: list of column headers for the csv. The column with header
                    'text' is preprocessed.
        returns: A list of preprocessed tokens for each input tweet.
        """
        df = pd.read_csv(csv, sep=',', header=None, names=columns)
        df['tokens'] = df['text'].apply(self.tokenize_tweet)
        return df['tokens'].values


if __name__ == '__main__':
    processor = Preprocessor()
    print(
        processor.tokenize_tweet(
            '@hereisauserhandle I <iamanhtmltag>  <3 summer!!!!! I got 0.55 dollars from nice-person@mail.com for visiting www.scam.com #happy'
        ))
