# TensorFlow_SimpsonsChatbot
A Chatbot based on Tensorflow, that answers to questions (resp. input sentences) with famous quotes from the Simpsons.
This chatbot is based on a [tutorial to contextual chatbots with tensorflow](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077) and hosts a REST-API to communicate with it.

## Build/Run with Docker
Run can train and run this Bot using docker. For this repeat the following steps (assuming your Docker environment is already set up):
* Change to the root directory of this project
* If you want to train the Bot with your own intents, just edit the [intents.json](data/intents.json) file but **keep the data structure as it is**
* Build the docker image: ```docker build -t simpsonsbot:latest .``` This will build the docker image including the download of all dependencies and training of the model based on the [intents.json](data/intents.json) file.
* Run the container: ```docker run --name simpsonsbot simpsonsbot:latest```

docker run -d -p 5000:5000 --name vagvag simpsonsbot:latest

## Run with pre-trained model
If you just want to experiment and use the model trained (training and model files available in this repo) you can just clone this repository, download all dependencies and execute the Flask REST-API via ```python -m flask run``` or ```python -m flask run --host=0.0.0.0``` if the REST-API should listen to external clients (not localhost only) as well.

## Examples
