from flask import Flask, request, render_template, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the chatbot model (GPT-Neo)
model_name = "EleutherAI/gpt-neo-1.3B"

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json['message']
    response = handle_input(user_input)
    return jsonify({'response': response})

def handle_input(user_input):
    response = gemma_2b_fine_tuned(user_input, max_length=100, do_sample=True, truncation=True)
    return response[0]['generated_text']

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

