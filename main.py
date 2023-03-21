from flask import *
from flask import jsonify, request, render_template

import processor
new= Flask(__name__)
new.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'
@new.route('/', methods=["GET", "POST"])


def hello_world():
    return render_template('index1.html', **locals()) # Running page


@new.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():

    if request.method == 'POST':
        the_question = request.form['question']

        response = processor.chatbot_response(the_question)

    return jsonify({"response": response })

if __name__ == '__main__':
    new.run()

