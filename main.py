#!/usr/bin/python3
################
#  PySemantic  #
################
# import seaborn as sns
# import pandas as pd
import word2vec
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import mysql.connector as mariadb
from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
import json as JSON
import re
import os

CONFIG_FILE = "config.json"

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

############
# Main App #
############
def main ():
    # Sets
    tfs_name, tfs_version = 'nnlm-en-dim128', 1
    # tfs_name = 'universal-sentence-encoder'

    # Get the configuration
    config = getConfig()

    # Connect to MariaDb
    print('* Connecting to MariaDb')
    # mysql_client = getClientMariaDb(config['mysql'])

    # Connect to MongoDb
    print('* Connecting to MongoDb')
    # mongodb_client = getClientMongoDb(config['mongodb'])

    # Initialize Tensorflow Hub
    print('* Load TensorflowHub.%s' % tfs_name)
    embed = hub.Module("https://tfhub.dev/google/%s/%s" % (tfs_name, tfs_version))

    # features = {
    #     'sentences': np.array([
    #         ['The', 'fish', 'swimming', 'in', 'the', 'water'],
    #         ['Sea', 'is', 'beautiful', '', '', '']
    #     ])
    # }
    #
    # embeddings = embed(tf.reshape(features["sentences"], shape=[-1]))

    words = ['cat', 'dog', 'fish']

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        embeddings = sess.run(embed(words))
        print(embeddings)

        for i, embeddings in enumerate(np.array(embeddings).tolist()):
            print("Message: {}".format(words[i]))
            print("Embedding size: {}".format(len(embeddings)))
            embeddings_snippet = ", ".join(
                (str(x) for x in embeddings[:3]))
            print("Embedding: [{}, ...]\n".format(embeddings_snippet))

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
