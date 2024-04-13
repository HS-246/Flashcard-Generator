from flask import Flask, request, jsonify
import main

app = Flask(__name__)

@app.route('/generate_flashcards', methods=['GET'])
def generate_flashcards():

    
    # Get the text input and number of flashcards from the request
    text = request.args.get('text')
    print(text)
    num_flashcards = request.args.get('num_flashcards_limit')
    print(num_flashcards)
    num_flashcards=int(num_flashcards)
    # Call your machine learning model to generate the flashcards
    flashcards = main.generate_flashcards(text, num_flashcards)

    # Return the flashcards as a JSON response
    response = {
        'flashcards': flashcards
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()