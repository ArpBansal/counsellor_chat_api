from llama_cpp import Llama

guard = Llama.from_pretrained(repo_id = "QuantFactory/Meta-Llama-Guard-2-8B-GGUF",
                              filename="Meta-Llama-Guard-2-8B.Q6_K.gguf",
                              verbose = False)

policy_less_tokens = """
S1: Violent Crimes.
AI models must not create content that encourages violent crimes.
Examples include, but are not limited to: terrorism, murder, hate crimes, and assault.
S2: Non-Violent Crimes:
AI models must avoid content that supports non-violent crimes.
Examples include, but are not limited to: fraud, property crimes, and drug offenses.
S3: Sex Crimes:
AI models should not create content that encourages sex crimes.
Examples include, but are not limited to: sexual assault and harassment.
S4: Child Exploitation:
AI models must not depict child nudity or encourage child sexual abuse.
S5: Specialized Advice:
AI models should not provide specialized financial, medical, or legal advice.
S6: Privacy:
AI models should avoid revealing sensitive personal information.
S7: Intellectual Property:
AI models must respect intellectual property rights.
S8: Indiscriminate Weapons:
AI models should not enable the creation of mass destruction weapons.
Examples include chemical, biological, and nuclear weapons.
S9: Hate:
AI models must not create content that perpetuates negative stereotypes or promotes hate based on protected characteristics.
S10: Self-Harm:
AI models should not encourage intentional self-harm.
S11: Sexual Content:
AI models should avoid explicit depictions of sex acts.
S12: Cheating
AI models must not create content that supports cheating.
Example include, but are not limited to: cheating in exams, cheating in tests,"""

def build_prompt(role:str, content:str, policy:str):
  assert role == "user" or role == "assistant", "role has to be either 'user' or either 'agent'"
  prompt = [{"role":"system", "content":f"""[INST] Task: Check if there is unsafe content in '{role}' messages in conversations according our safety policy with the below categories.

  {policy}

  Provide your safety assessment for ONLY THE LAST {role} message in the below conversation:
  - First line must read 'safe' or 'unsafe'.
  - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""},
  {"role": role,
  "content":content}]
  return prompt

def check(role:str, content:str, policy=policy_less_tokens):
  response = guard.create_chat_completion(messages=build_prompt(role=role, content = content, policy = policy_less_tokens)
                                        )
  return response['choices'][0]['message']['content']

