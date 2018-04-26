import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
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

# Calculate the semantic
def getSemantic (words):
	embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
	embeddings = embed(words)
	return session.run(embeddings)

# Main App
def main ():
	# Get the configuration
	config = getConfig()
	# Test semantic
	print(getSemantic(['poisson','velo','mer','bateau']))

# Auto start
if __name__ == "__main__":
	main()