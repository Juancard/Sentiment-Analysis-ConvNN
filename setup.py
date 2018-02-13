from distutils.core import setup

setup(name = "Sentiment-Analysis-ConvNN",
    version = "100",
    description = "Sentiment analysis using a convolutional neural network",
    author = "Juan Cardona",
    author_email = "juancruzcardona@gmail.com",
    url = "",
    packages = ['packages'],
    scripts = ["imdb1_main.py", "airline_tweets_main.py"]
) 
