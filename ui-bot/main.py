from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
model_name = "Ikeofai/gemma-2b-fine-tuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json['message']
    input_ids = torch.tensor(tokenizer.encode(user_input, return_tensors="pt")).to("cuda")
    outputs = model.generate(input_ids, max_length=1024, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0])
    return {"response": response}

if __name__ == "__main__":
    app.run(debug=True)
