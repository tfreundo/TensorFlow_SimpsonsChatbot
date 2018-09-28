FROM ubuntu:16.04
EXPOSE 5000
ADD . /bot

RUN apt-get update && apt-get -y install nano python3 python3-pip git
RUN pip3 install --upgrade numpy tensorflow tflearn nltk scipy flask-restful

# Install NLTK modules
WORKDIR /bot
RUN python3 nltk_modulesdownloader.py
RUN python3 bot_train.py
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]