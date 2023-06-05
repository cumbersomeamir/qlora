#pip3 install -U bitsandbytes scikit-learn
#pip3 install -U git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/accelerate.git git+https://github.com/huggingface/peft.git


from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Amirkid/Qlora"

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)



text = "Hello my name is"
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
