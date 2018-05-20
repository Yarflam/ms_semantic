# ms_semantic

Middle Service -  Semantic Analyzer for the project SEO.

## Prerequisite

You need a Docker environment to use it and two databases, MongoDB and MySQL.

- Mongodb is the hot base : the spider save the pages, you have a wonderfull datalack with to work.

- MySQL is the cold base : the script save the results, the front can show the data as they like.

# How to configure ?

Copy the file example.config.json and rename it as "config.json".

In the config.json, you have to add the credential informations for the databases.

## How to run the service ?

Execute on your command line :

> $ docker-compose up -d

It's run !

## How can I improve it ?

Simple, you have to remove the line "/bin/bash /home/pysemantic" in the entrypoint.

If you wan't to debugging the python script :

> $ docker exec pysemantic /bin/bash /home/pysemantic.sh

## How can I offer you a coffee ?

It's sweet. Thank you so much !

But I don't have any bitcoin address to available now. I will be back ...
