# Sentiment-Analysis-ConvNN
Sentiment analysis using a convolutional neural network
Report [here](https://docs.google.com/document/d/1sqN9UpXz3R2tUOsjIvW0Mr5gwIoae_fBpGJTZpmrtGw/edit?usp=sharing)

## Installation
1) Clone repo
```bash
git clone https://github.com/Juancard/Sentiment-Analysis-ConvNN.git
```
2) Install dependencies
```bash
pip install -r requirements.txt
```
3) Install modules of this app
```bash
python setup.py install
```
4) Rename `config.ini.example` to `config.ini`
In linux: 
```bash
cp config.ini.example config.ini
```
## Run
1) Modify `config.ini` with your preferred hyperparameters (don't forget setting Output path parameter of this file)
2) Execute:
```bash
python imdb1_main.py config.ini
```
or
```bash
python airway_tweets_main.py config.ini
```

