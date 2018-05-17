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
import urllib.request
import numpy as np
import json as JSON
import time
import re
import os

### CONSTANTS ###
LABELS_FILE = "labels"
CONFIG_FILE = "config.json"
MODELS_FOLDER = 'models'+os.sep

# TF ignore the standards errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Current path
dir_path = os.path.dirname(os.path.realpath(__file__))+os.sep

#########
# Tools #
#########

# Clean up any string chars (remove html, spaces and others)
def cleanString (code):
    code = re.sub('<script[^>]+>[^<]+</script>', ' ', code, re.IGNORECASE)
    code = re.sub('<[^>]+>', ' ', code)
    code = re.sub('[\\n\\r\\s;,: -]+', ' ', code)
    return code

# Open any files
def getFile (rel_path):
    # Check if the file exists
    if os.path.isfile('%s%s' % (dir_path, rel_path)):
        # Open the file
        with open('%s%s' % (dir_path, rel_path), 'r') as file:
            data = file.read()
        return data
    else:
        # Warning : the file doesn't exists
        print('Warning: the file %s%s doesn\'t exists.'
            % (dir_path, rel_path))
        quit()

# Get the content page of any url
def getUrlPage (url):
    data = b''
    with urllib.request.urlopen(url) as http:
        data = http.read()
    return data

# Download any file from the web
def download (url, path, name=False):
    if not '/' in url: return False
    if not os.path.exists('%s%s%s' % (dir_path, path, name)):
        # Extract the name of url
        pos = url.rindex('/')+1
        url, url_name = url[0:pos], url[pos:]
        if not len(url_name): url_name = 'index.html'
        name = url_name if not name else name
        # Start the download
        print('* Downloading %s%s' % (url, url_name))
        data = getUrlPage('%s%s' % (url, url_name))
        if len(data):
            # Save the data in the specific file
            with open('%s%s%s' % (dir_path, path, name), 'wb') as file:
                file.write(data)
        else: return False
    return True

# Bubble sort : combine keys and values
def bubbleSort (keys, values, byOrder=True):
    for i in range(0, len(values)):
        k, minmax = i, values[i]
        for j in range(i, len(values)):
            test = byOrder and values[j] < minmax
            test = test or (not byOrder and values[j] > minmax)
            if test: k, minmax = j, values[j]
        if k != i:
            keys[i], keys[k] = keys[k], keys[i]
            values[i], values[k] = values[k], values[i]
    return [keys, values]

# Configuration
def getConfig ():
    data = getFile(CONFIG_FILE)
    return JSON.loads(data)

# List of labels
def getLabels (model):
    data = getFile(LABELS_FILE)
    # Define the real labels
    labels = []
    for row in data.split('\n'):
        # Clean the words
        words = cleanString(row).lower().split(' ')
        # For each
        for word in words:
            # No duplicate label and if the word doesn't exists again
            if not word in labels and checkWordInModel(word, model):
                labels.append(word)
    return labels

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

# Download and load the model
def getFrWiki2Vec (fname):
    # Download the file
    download('%s%s' % ('http://embeddings.org/', fname), MODELS_FOLDER, fname)
    # Load the binary with Gensim (it's more efficient than Word2vec library)
    return KeyedVectors.load_word2vec_format('%s%s%s' % (dir_path, MODELS_FOLDER, fname), binary=True)

# If the word exists in the model
def checkWordInModel (word, model):
    return True if word in model.wv.vocab else False

# Transform all words to vectors
def wordsToVect (words, model, op_wlen=False, op_distinct=False):
    i, vectors, wlen = 0, [], []
    if type(words) is str: words = words.split()
    if type(words) is not list: return vectors
    # For each words
    while i < len(words):
        w = words[i]
        if type(w) is str:
            if chr(32) in w:
                # It's a sentence : split and continue
                words += w.split()
            else:
                # If the word is in the model
                if checkWordInModel(w, model):
                    # Get the vector
                    vector = np.array(model.get_vector(w))
                    # Apply the distinct mode
                    if not vector in vectors or not op_distinct:
                        vectors.append( vector )
                        wlen.append( len(w) )
        # Next word
        i += 1
    # Apply the wlen mode : add the priority of the word
    return [vectors, wlen] if op_wlen else vectors

# Calculate the central point of multiple vectors
def getCPointVect (vectors, wlen):
    if type(vectors) is not list or type(wlen) is not list: return np.array([])
    if not len(vectors) and not len(vectors) == len(wlen): return np.array([])
    cpoint = vectors[0]
    # For each vectors
    for i in range(1, len(vectors)):
        # The vectors are multiply and added
        cpoint += vectors[i] * np.int(wlen[i])
    cpoint /= np.int(len(vectors))
    return cpoint

# Vector to word
def vectToWord (vector, model):
    keys, values = [], []
    kwords = model.similar_by_vector(vector)
    if not len(kwords): return ''
    # Separate words and ratio
    for kword in kwords:
        keys.append(kword[0])
        values.append(kword[1])
    # Sort and return the greater
    keys, values = bubbleSort(keys, values, False)
    return keys[0]

# Get the most predictible label
def getBetterLabel (features, labels, model):
    if type(labels) is not list or not len(labels): return ''
    # Feature or Features
    if type(features) is list:
        # We have features
        keys, values = [], []
        for feature in features:
            label = getBetterLabel(feature, labels, model)
            if not label in keys:
                keys.append(label)
                values.append(0)
            values[keys.index(label)] += 1
        # Sort by the most popular
        keys, values = bubbleSort(keys, values, False)
        return keys[0]
    elif type(features) is str:
        # For each label
        j, min = 0, model.distance(features, labels[0])
        for i in range(1, len(labels)):
            # Get the distance with the features
            dist = abs(model.distance(features, labels[i]))
            # And keep the smaller distance
            if dist < min:
                j, min = i, dist
        return labels[j]
    else:
        return ''

# Combine all predictables labels and get the better
def vectToBetterLabel (vector, labels, model):
    if type(labels) is not list or not len(labels): return ''
    # Extract the similar words
    keys, values = [], []
    kwords = model.similar_by_vector(vector)
    # Estimate the betters labels
    for kword in kwords:
        label = getBetterLabel(kword[0], labels, model)
        if not label in keys:
            keys.append(label)
            values.append(0)
        values[keys.index(label)] += 1
    # Sort by the most popular
    keys, values = bubbleSort(keys, values, False)
    return keys[0]

############
# Main App #
############
def main ():
    # Sets
    # tfs_name = 'universal-sentence-encoder'
    # tfs_name, tfs_version = 'nnlm-en-dim128', 1
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

    # Get the labels
    print('* Get the labels')
    labels = getLabels(model)
    print('->', labels)

    # Test
    items = [
        {
            'html': getUrlPage('https://www.onedayonetravel.com/blog-voyage-voyage-en-thailande/').decode()
        }
    ]

    # Begin the treatment
    print('* Get the last documents')
    # items = getLastSpiderResults(mongodb_client, 1)
    if len(items):
        # Get the fist document
        html = items[0]['html']
        # Get the words
        i, words = 0, list(set(cleanString(html).lower().split()))
        # Extract the exists words
        while i < len(words):
            if not checkWordInModel(words[i], model):
                words = words[0:i] + words[i+1:]
            else: i += 1
        print(words)
        # Extract the theme
        theme = getBetterLabel(words, labels, model)
        print(theme)

    print('End')

# Auto start
if __name__ == "__main__":
    main()
