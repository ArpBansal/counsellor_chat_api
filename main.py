# from flask import Flask, request, jsonify


# app = Flask(__name__)

# @app.route('/')
# def index():
#     return "Hello, this is the AI chat API for Ollama."

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.json
#     user_message = data.get('message')
#     # integrate your AI model to generate a response
#     ai_response = generate_response(user_message)
#     return jsonify({"response": ai_response})

# def generate_response(user_message):
#     # Placeholder function for AI model integration
#     return f"Echo: {user_message}"

# if __name__ == '__main__':
#     app.run(debug=True)

# fastapi_implementation
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import ChatRequest
from cookbook.llm import llm_function
from cookbook.llama_guard import check

app=FastAPI()

@app.post("/counsellor")
def counsellor(quest:str):
    if check(role="user", content=quest) == 'safe':
        respon = llm_function(user_input=quest)
        return respon
    else:
        return "invalid_request"