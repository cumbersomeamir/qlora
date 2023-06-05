#pip3 install -U bitsandbytes scikit-learn
#pip3 install -U git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/accelerate.git git+https://github.com/huggingface/peft.git


import os
import bitsandbytes as bnb
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
peft_model_id = "Amirkid/Qlora"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_4bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

prompts = ["Amir opened up with this joke",
           "The professor started the lecture by saying",
           "The captain began the journey by",
           "The chef surprised everyone by",
           "In the beginning, the novel",
           "The concert began with the song",
           "She kicked off the meeting by",
           "The game started when",
           "The trip began with a surprise",
           "He opened the letter and"]

for prompt in prompts:
    batch = tokenizer(prompt, return_tensors='pt')

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=50, num_return_sequences=10, do_sample=True)

    for i, output_token in enumerate(output_tokens):
        print(f"\nPrompt: {prompt}\nCompletion {i+1}: {tokenizer.decode(output_token, skip_special_tokens=True)}")
