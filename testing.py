import os
import bitsandbytes as bnb
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"]="0"
peft_model_id = "Amirkid/Qlora"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_4bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

# Loop to keep the conversation going
while True:
    # Accept user input
    user_input = input("You: ")

    # If the user types 'exit', end the conversation
    if user_input.lower() == 'exit':
        break

    batch = tokenizer(user_input, return_tensors='pt')
    
    # Remove 'token_type_ids' from the batch
    if 'token_type_ids' in batch:
        del batch['token_type_ids']

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=50)

    print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
