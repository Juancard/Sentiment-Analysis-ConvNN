import configparser
import logging


def setLogger():
	"""
	To print logs
	"""
	logging.basicConfig(level=logging.DEBUG)
	logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
def loadConfigData(iniFilePath):
	"""
	to load configuration data from .ini file
	"""
	config = configparser.ConfigParser()
	config.read(iniFilePath)
	config = {
		'coll_path': config['COLLECTION']['PATH'],
		'sentence_length': int(config['MODEL']['SENTENCE_LENGTH']),
		'vocab_size': int(config['EMBEDDING']['VOCABULARY_SIZE']),
		'embed_length': int(config['EMBEDDING']['VECTOR_LENGTH']),
		'embed_pretrained': int(config['EMBEDDING']['PRETRAINED']),
		'embed_pretrained_path': config['EMBEDDING']['PRETRAINED_PATH'],
		'embed_train': int(config['EMBEDDING']['TRAIN']),
		'filters': int(config['MODEL']['FILTERS']),
		'filter_sizes': [int(fs) for fs in config['MODEL']['FILTER_SIZES'].split(',')],
		'epochs': int(config['FITTING']['EPOCHS']),
		'batch_size': int(config['FITTING']['BATCH_SIZE']),
		'dropout_prop': float(config['MODEL']['DROPOUT']),
		'loss_func': config['MODEL']['LOSS_FUNCTION'],
		'optimizer': config['MODEL']['OPTIMIZER'],
		'output_path': config['OUTPUT']['PATH']
	}
	return config


