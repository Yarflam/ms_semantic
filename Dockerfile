FROM tensorflow/tensorflow:latest-py3
MAINTAINER Yarflam

# Install standards dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl wget nano apt-transport-https lsb-release ca-certificates git

# Install TK
RUN apt-get install -y --no-install-recommends python3-tk

# Upgrade PIP
RUN pip install --upgrade pip

# Install Tensorflow Hub
RUN pip install tensorflow-hub

# Install the MySQL connector
RUN apt-get update && apt-get install -y --no-install-recommends libmysqlclient-dev
RUN pip install mysql-connector

# Install the MongoDb connector
RUN pip install pymongo

# Install french pack
RUN apt-get install -y --no-install-recommends locales &&\
    locale-gen en_CA.UTF-8 &&\
    echo "environment=LANG=\"es_ES.utf8\", LC_ALL=\"es_ES.UTF-8\", LC_LANG=\"es_ES.UTF-8\"" > /etc/default/locale
RUN chmod 0755 /etc/default/locale
ENV LC_ALL=en_CA.UTF-8
ENV LANG=en_CA.UTF-8
ENV LANGUAGE=en_CA.UTF-8

# Install Word2vec
RUN pip install cython
RUN pip install word2vec

# Install Gensim
RUN pip install gensim

# Launcher python script
RUN echo "#!/bin/bash" > /home/pysemantic.sh &&\
    echo "export LANG=en_US.UTF-8" >> /home/pysemantic.sh &&\
    echo "cd /notebooks/semantic" >> /home/pysemantic.sh &&\
    echo "python -u main.py" >> /home/pysemantic.sh

RUN echo "#!/bin/bash" > /home/entrypoint.sh &&\
    echo "git clone https://github.com/Yarflam/ms_semantic /notebooks/semantic" >> /home/entrypoint.sh &&\
    echo "while [[ 1 ]]; do" >> /home/entrypoint.sh &&\
#    echo "/bin/bash /home/pysemantic" >> /home/entrypoint.sh &&\
    echo "sleep 10" >> /home/entrypoint.sh &&\
    echo "done" >> /home/entrypoint.sh

CMD [ "/bin/bash", "/home/entrypoint.sh" ]
