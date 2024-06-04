
# from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
# from langchain.chains import LLMChain, ConversationChain#, conversational_retrieval
# from fastapi import FastAPI

from llama_cpp import Llama

system_prompt = """You are a excellent counsellor that helps learner with their mental health, their obstacles in education and their day-to-day life problems
                user will ask you questions and you will carefully answer them"""
B_INST, E_INST = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>", "<|eot_id|>"
B_SYS, E_SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|>"
ASSISTANT_INST = "<|start_header_id|>assistant<|end_header_id|>"
SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS

def prompt_for_chat(content:str):
  return [{"role": "system", "content": """You are an excellent counselor who assists user with their mental health,
            educational challenges, and everyday life issues.
            user will ask you questions, and you will provide thoughtful answers."""},
          {"role": "user",
           "content":content}]

model = Llama.from_pretrained(repo_id="Arpit-Bansal/counsellor_model_q5_k_m",
                              filename="counsellor_model_q5_k_m-unsloth.Q5_K_M.gguf",
                              verbose=False)

def prompt_for_chat(content:str):
  return [{"role": "system", "content": """You are an excellent counselor who assists user with their mental health,
            educational challenges, and everyday life issues.
            and you will provide thoughtful answers.""",
          },
          { "role": "user",
           "content":f"{content}{E_INST}"}]

def response_return(response):
  res = ""
  for chunk in response:
    delta = chunk["choices"][0]["delta"]
    if "content" not in delta:
        continue
    res += delta["content"]
  return res

# memory= ConversationBufferWindowMemory(input_key="question", memory_key="chat_history", ai_prefix="assistant", human_prefix="user", k=4)

def llm_function(user_input:str):
    # chat_history = memory.load_memory_variables({})["chat_history"]
    # prompt = f"{SYSTEM_PROMPT}\n\n{chat_history}\n{B_INST} {user_input} {E_INST}"
    print(prompt_for_chat(content=user_input))
    llm_response = model.create_chat_completion(messages=prompt_for_chat(content=user_input),
                                        stream = True, temperature = 0.6, max_tokens = 256)
    resp = response_return(llm_response)
    # memory.chat_memory.add_user_message(user_input)
    # memory.chat_memory.add_ai_message(resp)
    return resp
#llama-guard-2

# app = FastAPI()



# @app.post("/")
# async def stream(item:ChatRequest):
#     return llm_function(item)