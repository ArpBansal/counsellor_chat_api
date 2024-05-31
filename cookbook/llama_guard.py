import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from ctransformers import AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-Guard-2-8B"
# device = "cuda"
guard = AutoModelForCausalLM.from_pretrained("QuantFactory/Meta-Llama-Guard-2-8B-GGUF",
                                             model_type = "llama",
                                             model_file="Meta-Llama-Guard-2-8B.Q5_1.gguf")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=ACCESS_TOKEN)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

llm_guard = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # device_map=device,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    token=ACCESS_TOKEN,
)
policy = """S1: Violent Crimes.
AI models should not create content that enables, encourages, or excuses the commission of violent crimes. Examples of violent crimes include, but are not limited to:
 - Unlawful violence toward people (ex: terrorism, genocide, murder, hate crimes, child abuse, assault, battery, kidnapping)
 - Unlawful violence toward animals (ex: animal abuse)
S2: Non-Violent Crimes.
AI models should not create content that enables, encourages, or excuses the commission of non-violent crimes. Examples of non-violent crimes include, but are not limited to:
 - Financial crimes (ex: fraud, scams, money laundering)
 - Property crimes (ex: burglary, robbery, arson, vandalism)
 - Drug crimes (ex: creating or trafficking narcotics)
 - Weapons crimes (ex: producing unlicensed firearms)
 - Cyber crimes (ex: hacking, spyware, malware)
S3: Sex Crimes.
AI models should not create content that enables, encourages, or excuses the commission of sex crimes. Examples of sex crimes include, but are not limited to:
 - Human trafficking
 - Sexual assault (ex: rape)
 - Sexual harassment (ex: groping)
 - Lewd conduct
S4: Child Exploitation.
AI models should not create content that depicts child nudity or that enables, encourages, excuses, or depicts the sexual abuse of children.
S5: Specialized Advice.
AI models should not create content that contains specialized financial, medical, or legal advice.
S6: Privacy.
AI models should not create content that contains sensitive, personal information about private individuals.
S7: Intellectual Property.
AI models should not create content that violates the intellectual property rights of third parties.
S8: Indiscriminate Weapons.
AI models should not create content that enables the creation of weapons of mass destruction. Examples of such weapons include, but are not limited to:
 - Chemical weapons (ex: nerve gas)
 - Biological weapons (ex: anthrax)
 - Radiological weapons (ex: salted bombs)
 - Nuclear weapons (ex: atomic warheads)
 - High-yield explosive weapons (ex: cluster munitions)
S9 Hate.
AI models should not create content that is hateful toward people on the basis of their protected characteristics (race, color, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity and serious disease). AI models should not create content that perpetuates negative stereotypes.
S10: Self-Harm.
AI models should not create content that enables, encourages, or excuses acts of intentional self-harm. Examples of such actions include, but are not limited to:
 - Suicide
 - Self-injury (ex: cutting)
 - Disordered eating
S11: Sexual Content.
AI models should not create depictions of nude adults or content that contains erotic descriptions or explicit depictions of sex acts."""

def build_prompt(role:str, content:str, policy:str):
  assert role == "user" or role == "assistant", "role has to be either 'user' or either 'agent'"
  prompt = f"""[INST] Task: Check if there is unsafe content in '{role}' messages in conversations according our safety policy with the below categories.


  {policy}




  {role}: {content}



  Provide your safety assessment for ONLY THE LAST {role} message in the above conversation:
  - First line must read 'safe' or 'unsafe'.
  - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""
  return prompt

def check(role:str, content:str, policy:str, verbose=False):
  prompt = build_prompt(role=role, content=content, policy=policy)
  if verbose:
    print(prompt)


    output = guard(prompt=prompt, max_new_tokens=100)
    

  return output

def parser(message: str):
    result = {"safe": False, "categories": ""}
    try:
        lines = message.splitlines()
        if lines[0] == "safe":
            result["safe"] = True
        else:
            result["categories"] = lines[1]
    except Exception as e:
        print(f"bad model output: {e}")
    return result

