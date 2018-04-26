import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import mysql.connector as mariadb
# import seaborn as sns
# import pandas as pd
import numpy as np
import json as JSON
import re
import os

CONFIG_FILE = "config.json"

# TF ignore the standards errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
def getConfig ():
	# Open the config file
	with open(CONFIG_FILE, 'r') as file:
		# Read and extract it
		data = file.read()
		json = JSON.loads(data)
	return json

# Main App
def main ():
	# Get the configuration
	config = getConfig()

	# Connect to MariaDb
	mysql_config = config['mysql']
	mysql_con = mariadb.connect(
		host = mysql_config['host'],
		user = mysql_config['user'],
		password = mysql_config['pass'],
		database = mysql_config['database'])
	dbHandler = mysql_con.cursor()
	dbHandler.execute('SHOW TABLES')
	result = dbHandler.fetchAll()
	for item in result:
		print(item)

	print('End')

# Auto start
if __name__ == "__main__":
	main()