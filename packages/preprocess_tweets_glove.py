"""
python preprocess_tweets_glove.py "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"
Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu
Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
Added function 'clean_str' by juancard adapted from Yoon repo:
https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
"""

import sys
import re

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    # Split hashtags on uppercase letters
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps> "


def tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ")
    text = re_sub(r"@\w+", " <user> ")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " <smile> ")
    text = re_sub(r"{}{}p+".format(eyes, nose), " <lolface> ")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " <sadface> ")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " <neutralface> ")
    text = re_sub(r"<3"," <heart> ")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ")
    text = re_sub(r"#\S+", hashtag)
    # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
    text = re_sub(r"([!?.]){2,}", r" \1 <repeat> ")
    # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r" \1\2 <elong> ")
    # Force splitting words appended with slashes (once we tokenized the URLs, of course)
    text = re_sub(r"/"," / ")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, string, flags=FLAGS)

    string = re_sub(r"[^A-Za-z0-9(),!?\'\`<>]", " ")
    string = re_sub(r"\'s", " \'s")
    string = re_sub(r"\'ve", " \'ve")
    string = re_sub(r"n\'t", " n\'t")
    string = re_sub(r"\'re", " \'re")
    string = re_sub(r"\'d", " \'d")
    string = re_sub(r"\'ll", " \'ll")
    string = re_sub(r",", " , ")
    string = re_sub(r"!", " ! ")
    string = re_sub(r"\(", " ( ")
    string = re_sub(r"\)", " ) ")
    string = re_sub(r"\?", " ? ")
    string = re_sub(r"\s{2,}", " ")
    return string.strip().lower()

if __name__ == '__main__':
    _, text = sys.argv
    if text == "test":
        text = "I TEST alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
    tokens = tokenize(text)
    tokens = clean_str(tokens)
    print tokens
