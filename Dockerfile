FROM ubuntu:16.04
ADD . /bot

RUN apt-get update && apt-get -y install nano python3 python3-pip
RUN pip3 install --upgrade numpy tensorflow tflearn nltk scipy flask-restful

# Install NLTK modules
WORKDIR /bot
RUN python3 nltk_modulesdownloader.py
RUN python3 bot.py train
CMD ["python3", "-m", "flask", "run"]