#!/usr/bin/python3
################
#  PySemantic  #
################
# import seaborn as sns
# import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import mysql.connector as mariadb
from pymongo import MongoClient
from bson.json_util import dumps
from bson.objectid import ObjectId
import numpy as np
import json as JSON
import re
import os

CONFIG_FILE = "config.json"

# TF ignore the standards errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#########
# Tools #
#########

def cleanUpUnicode (text):
	matches = re.finditer("\\\\u([a-z0-9]{2})([a-z0-9]{2})", text)
	for match in matches:
		cdec = int(match.group(2), 16)
		if cdec >= 128:
			chex = hex(cdec % 128)[2:].zfill(2)
			text = text.replace(match.group(0), '')
	return text

def getKeywords (code):
	code = re.sub('<script[^>]+>[^<]+</script>', ' ', code, re.IGNORECASE)
	code = re.sub('<[^>]+>', ' ', code)
	code = re.sub('[\\n\\r\\s ]+', ' ', code)
	return code

# Configuration
def getConfig ():
	# Open the config file
	with open(CONFIG_FILE, 'r') as file:
		# Read and extract it
		data = file.read()
		json = JSON.loads(data)
	return json

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
		output.append(JSON.loads(dumps(doc)))
	return output

############
# Main App #
############
def main ():
	# Get the configuration
	config = getConfig()

	# Connect to MariaDb
	# mysql_client = getClientMariaDb(config['mysql'])

	# Connect to MongoDb
	mongodb_client = getClientMongoDb(config['mongodb'])

	#
	# Begin the treatment
	#
	items = getLastSpiderResults(mongodb_client, 1)
	if len(items):
		# Get the fist document
		html = items[0]['html']
		# Get the keywords
		keywords = getKeywords(html)
		# test = cleanUpUnicode(keywords)
		print(keywords)

	print('End')

# Auto start
if __name__ == "__main__":
	main()