# Sentiment-Analysis-ConvNN
Sentiment analysis using a convolutional neural network

## Installation
1) Clone repo
git clone https://github.com/Juancard/Sentiment-Analysis-ConvNN.git

2) Install dependencies
pip install -r requirements.txt

3) Install modules of this app
python setup.py install

4) Rename `config.ini.example` to `config.ini`
In linux: cp config.ini.example config.ini

## Run
1) Modify `config.ini` with your preferred hyperparameters (don't forget setting Output path parameter of this file)
2) Execute:
python imdb1_main.py config.ini
