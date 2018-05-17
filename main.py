#!/usr/bin/python3
################
#  PySemantic  #
################
# import seaborn as sns
# import pandas as pd

# Semantic Fields
import word2vec
from gensim.models import KeyedVectors

# Machine Learning
import tensorflow as tf
import tensorflow_hub as hub

# Graphic Tools
import matplotlib.pyplot as plt

# DB Connectors
import mysql.connector as mariadb
from pymongo import MongoClient

# Utilities
from bson.objectid import ObjectId
import urllib.request
import numpy as np
import json as JSON
import time
import re
import os

CONFIG_FILE = "config.json"
MODELS_FOLDER = './models/'

# TF ignore the standards errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Current path
dir_path = os.path.dirname(os.path.realpath(__file__))+os.sep

#########
# Tools #
#########

def getExtractWords (code):
    code = re.sub('<script[^>]+>[^<]+</script>', ' ', code, re.IGNORECASE)
    code = re.sub('<[^>]+>', ' ', code)
    code = re.sub('[\\n\\r\\s;,: -]+', ' ', code)
    return code

# Configuration
def getConfig ():
    # Check the config file
    if os.path.isfile(dir_path+CONFIG_FILE):
        # Open the config file
        with open(dir_path+CONFIG_FILE, 'r') as file:
            # Read and extract it
            data = file.read()
            json = JSON.loads(data)
        return json
    else:
        print('Warning: the configuration file %s doesn\'t exists.'
            % (dir_path+CONFIG_FILE))
        quit()

##############
# DB Clients #
##############

def getClientMariaDb (args):
    return mariadb.connect(
        host = args['host'],
        port = args['port'],
        user = args['user'],
        password = args['pass'],
        database = args['base'])

def getClientMongoDb (args):
    return MongoClient('mongodb://%s:%s@%s:%s/%s' % (
        args['user'], args['pass'], args['host'],
        args['port'], args['base']
    ))

##############
# DB Request #
##############

def queryMariaDb (mysql_client, request):
    dbHandle = mysql_client.cursor()
    dbHandle.execute(request)
    return dbHandle.fetchall()

##########################
# Specific Data Endpoint #
##########################

def getLastSpiderResults (mongodb_client, n=10):
    output = []
    cursor = mongodb_client.admin.spider_results.find({'ms_semantic': False}).limit(n)
    for doc in cursor:
        output.append(doc)
    return output

##################
# Models Helpers #
##################

def getFrWiki2Vec (binName):
    binSource = 'http://embeddings.org/'
    # Download
    if not os.path.exists('%s%s' % (MODELS_FOLDER, binName)):
        print('* Downloading %s%s' % (binSource, binName))
        with urllib.request.urlopen('%s%s' % (binSource, binName)) as http:
            binData = http.read()
            if len(binData):
                with open('%s%s' % (MODELS_FOLDER, binName), 'wb') as file:
                    file.write(binData)
            else:
                raise ValueError('The data file ')
            del binData
    # Load the binary with Gensim (it's most efficient than Word2vec library)
    return KeyedVectors.load_word2vec_format('%s%s' % (MODELS_FOLDER, binName), binary=True)

def checkWordsInModel (word, model):
    return True if word in model.wv.vocab else False

def wordsToVect (words, model):
    vectors = []
    if type(words) is str: words = words.split()
    if type(words) is not list: return vectors
    # For each words
    i = 0
    while i < len(words):
        w = words[i]
        if type(w) is str:
            if chr(32) in w:
                # If we have a sentence : split and continue
                words += w.split()
            else:
                # If the word is in the model
                if checkWordsInModel(w, model):
                    vectors.append( np.array(model.get_vector(w)) )
        i += 1
    return vectors

############
# Main App #
############
def main ():
    # Sets
    tfs_name, tfs_version = 'nnlm-en-dim128', 1
    # tfs_name = 'universal-sentence-encoder'
    # wcbdd_name = 'frWac_no_postag_no_phrase_700_skip_cut50.bin';
    wcbdd_name = 'frWiki_no_lem_no_postag_no_phrase_1000_cbow_cut100.bin'

    # Get the configuration
    config = getConfig()

    # Connect to MariaDb
    print('* Connecting to MariaDb')
    # mysql_client = getClientMariaDb(config['mysql'])

    # Connect to MongoDb
    print('* Connecting to MongoDb')
    # mongodb_client = getClientMongoDb(config['mongodb'])

    # Initialize Word2vec with French Data Model
    print('* Loading Word2vec')
    ts_begin = time.time()
    model = getFrWiki2Vec(wcbdd_name)
    print('* Word2vec is loaded ('+str(int(time.time() - ts_begin))+' s)')

    sentence = wordsToVect('une baleine dans la mer bleue', model);
    print(sentence, len(sentence))

    sim = model.similar_by_vector(sentence[1])
    print(sim)

    return True

    # Begin the treatment
    print('* Get the last documents')
    items = getLastSpiderResults(mongodb_client, 1)
    if len(items):
        # Get the fist document
        html = items[0]['html']
        # Get the words
        words = getExtractWords(html)

    print('End')

# Auto start
if __name__ == "__main__":
    main()
