from ..schemas import ChatRequest

from langchain.llms import ctransformers
from langchain.memory import ConversationBufferMemory #, ConversationSummaryMemory, ConversationSummaryBufferMemory
from langchain.chains.llm import LLMChain
from ctransformers import AutoModelForCausalLM
from fastapi import FastAPI
from llama_guard import guard, parser, build_prompt, check


system_prompt = """You are a excellent counsellor that helps learner with their mental health, their obstacles in education and their day-to-day life problems
                user will ask you questions and you will carefully answer them"""
B_INST, E_INST = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>", "<|eot_id|>"
B_SYS, E_SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|>"
ASSISTANT_INST = "<|start_header_id|>assistant<|end_header_id|>"
SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS


llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id="hf_repo_id", model_type='llama')
memory  = ConversationBufferMemory(input_key="question", memory_key="chat_history")

def llm_function(user_input):
    memory.chat_memory.add_user_message(user_input)
    chat_history = memory.load_memory_variables({})["chat_history"]
    prompt = f"{SYSTEM_PROMPT}\n\n{chat_history}\n{B_INST} {user_input} {E_INST}"
    llm_response = llm(prompt=prompt, temperature=0.7, repetition_penalty=1.2, threads=3, max_new_tokens=2048)
    memory.chat_memory.add_ai_message(llm_response)

    return llm_response
    
#llama-guard-2

app = FastAPI()


@app.post("/")
async def stream(item:ChatRequest):
    return llm_function(item)