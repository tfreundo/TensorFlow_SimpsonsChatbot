from flask import Flask, request
from bot import Bot

app = Flask(__name__)
bot = Bot()

print(bot.get_stats())

@app.route('/')
def chatbot():
    
    input_sentence = request.args.get('input_sentence')
    # TODO Readd as soon as context is necessary
    #user_id = request.args.get('user_id')

    print("Received input: ", input_sentence)
    response = bot.ask(input_sentence, 123)
    return response


