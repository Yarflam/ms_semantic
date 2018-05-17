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
LABELS_FILE = "labels"
MODELS_FOLDER = 'models'+os.sep

# TF ignore the standards errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Current path
dir_path = os.path.dirname(os.path.realpath(__file__))+os.sep

#########
# Tools #
#########

def cleanString (code):
    code = re.sub('<script[^>]+>[^<]+</script>', ' ', code, re.IGNORECASE)
    code = re.sub('<[^>]+>', ' ', code)
    code = re.sub('[\\n\\r\\s;,: -]+', ' ', code)
    return code

# Configuration
def getConfig ():
    # Check the config file
    if os.path.isfile('%s%s' % (dir_path, CONFIG_FILE)):
        # Open the config file
        with open('%s%s' % (dir_path, CONFIG_FILE), 'r') as file:
            # Read and extract it
            data = file.read()
            json = JSON.loads(data)
        return json
    else:
        print('Warning: the configuration file %s%s doesn\'t exists.'
            % (dir_path, CONFIG_FILE))
        quit()

# List of labels
def getLabels (model):
    # Check the labels file
    if os.path.isfile('%s%s' % (dir_path, LABELS_FILE)):
        # Open the labels file
        with open('%s%s' % (dir_path, LABELS_FILE), 'r') as file:
            data = file.read()
        # Proper data
        labels = []
        for row in data.split('\n'):
            words = cleanString(row).lower().split(' ')
            for word in words:
                if checkWordInModel(word, model):
                    labels.append(word)
        return labels
    else:
        print('Warning: the label file %s%s doesn\'t exists.'
            % (dir_path, LABELS_FILE))
        quit()

# Return the url's content
def getUrlPage (url):
    data = b''
    with urllib.request.urlopen(url) as http:
        data = http.read()
    return data

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
def getFrWiki2Vec (binName):
    binSource = 'http://embeddings.org/'
    # Download
    if not os.path.exists('%s%s%s' % (dir_path, MODELS_FOLDER, binName)):
        print('* Downloading %s%s' % (binSource, binName))
        binData = getUrlPage('%s%s' % (binSource, binName))
        if len(binData):
            with open('%s%s%s' % (dir_path, MODELS_FOLDER, binName), 'wb') as file:
                file.write(binData)
        else:
            raise ValueError('The data file ')
        del binData
    # Load the binary with Gensim (it's most efficient than Word2vec library)
    return KeyedVectors.load_word2vec_format('%s%s%s' % (dir_path, MODELS_FOLDER, binName), binary=True)

def checkWordInModel (word, model):
    return True if word in model.wv.vocab else False

# Transform all words to vectors
def wordsToVect (words, model, op_wlen=True):
    vectors, wlen = [], []
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
                if checkWordInModel(w, model):
                    vectors.append( np.array(model.get_vector(w)) )
                    wlen.append( len(w) )
        i += 1
    return [vectors, wlen] if op_wlen else vectors

# Calculate the central point of multiple vectors
def getCPointVect (vectors, wlen):
    if type(vectors) is not list or type(wlen) is not list: return np.array([])
    if not len(vectors) and not len(vectors) == len(wlen): return np.array([])
    cpoint = vectors[0]
    # For each vectors
    for i in range(1, len(vectors)):
        cpoint += vectors[i] * np.int(wlen[i])
    cpoint /= np.int(len(vectors))
    return cpoint

# Vector to word
def vectToWord (vector, model):
    words = model.similar_by_vector(vector)
    return words[len(words)-1][0] if len(words) else ''

# Get the most predictable label
def getBetterLabel (features, labels, model):
    if type(labels) is not list or not len(labels): return ''
    # Feature or Features
    if type(features) is list:
        # We have features
        kLabels, vLabels = [], []
        for feature in features:
            label = getBetterLabel(feature, labels, model)
            if not label in kLabels:
                kLabels.append(label)
                vLabels.append(0)
            vLabels[kLabels.index(label)] += 1
        # Sort by the most popular
        for i in range(0, len(vLabels)):
            index, mx = i, vLabels[i]
            for j in range(i, len(vLabels)):
                if vLabels[j] > mx:
                    index, mx = j, vLabels[j]
            if not index == i:
                kLabels[i], kLabels[index] = kLabels[index], kLabels[i]
                vLabels[i], vLabels[index] = vLabels[index], vLabels[i]
        return kLabels[0]
    elif type(features) is str:
        # For each label
        index, min = 0, model.distance(features, labels[0])
        for i in range(1, len(labels)):
            # Get the distance with the features
            dist = abs(model.distance(features, labels[i]))
            # And keep the smaller distance
            if dist < min:
                index, min = i, dist
        return labels[index]
    else:
        return ''

# Combine all predictables labels and get the better
def vectToBetterLabel (vector, labels, model):
    if type(labels) is not list or not len(labels): return ''
    # Extract the similar words
    kLabels, vLabels = [], []
    items = model.similar_by_vector(vector)
    # Estimate the betters labels
    for item in items:
        label = getBetterLabel(item[0], labels, model)
        if not label in kLabels:
            kLabels.append(label)
            vLabels.append(0)
        vLabels[kLabels.index(label)] += 1
    # Sort by the most popular
    for i in range(0, len(vLabels)):
        index, mx = i, vLabels[i]
        for j in range(i, len(vLabels)):
            if vLabels[j] > mx:
                index, mx = j, vLabels[j]
        if not index == i:
            kLabels[i], kLabels[index] = kLabels[index], kLabels[i]
            vLabels[i], vLabels[index] = vLabels[index], vLabels[i]
    return kLabels[0]

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

    # Begin the treatment
    print('* Get the last documents')
    # items = getLastSpiderResults(mongodb_client, 1)
    items = [
        {
            'html': getUrlPage('https://www.onedayonetravel.com/blog-voyage-voyage-en-thailande/').decode()
        }
    ]
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
