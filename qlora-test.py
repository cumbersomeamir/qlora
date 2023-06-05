#pip3 install -U bitandbytes datasets 
#pip3 install -U git+https://github.com/huggingface/accelerate.git git+https://github.com/huggingface/peft.git git+https://github.com/huggingface/transformers.git 


#Importing the relevant Libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import transformers
from datasets import load_dataset


#Loading the model and defining the BitsAndBytes Config
model_id = "EleutherAI/gpt-neox-20b"
bnb_config = BitsAndBytesConfig(load_in_4bit = True,
                                bnb_4bit_use_double_quant = True,
                                bnb_4bit_quant_type = "nf4"
                                bnb_4bit_compute_dtype = torch.bfloat16
                                )
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config, device_map = {"": 0})

#Enabling Gradient Checkpointing
model.gradient_checkpoiting_enable()
model = prepare_model_for_kbit_traning(model)

#Printing the Total Number of trainable parameters
def print_trainable_parameters(model):

  trainable_params = 0 
  all_param = 0

  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()

  print(f"trainable params:{trainable_params} || all parameters: {all_param} || trainable % : {100*trainable_params/all_param}")


#Defining the Lora Config
config = LoraConfig(r = 8,
                    lora_alpha = 32,
                    target_modules = ['query_key_value']
                    lora_dropout = 0.05,
                    bias = "none",
                    task_type = "CAUSAL_LM"
                    )

#Converting model to peft model
model = get_peft_model(model, config)
#Calling the print trainable parameters function
print_trainable_parameters(model)

#Loading the Dataset
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples : tokenizer(samples["quote"]), batched = True)


#Adding the pad token 
tokenizer.pad_token = tokenizer.eos_token

#Initialising the Trainer
trainer = transformers.Trainer(
    model = model
    train_dataset = data["train"]
    args = transformers.TrainingArguments(
        per_device_batch_size =1,
        gradient_accumulation_steps= 4,
        warmup_steps = 2,
        max_steps = 10,
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 1,
        output_dir = "outputs"
        optim = "page_adamw_8bit"
    )
    data_collator = transformers.DataCollatorForLanguageModelling(tokenizer, mlm = False)
    )
#Use cache or not
model.config.use_cache = False
#Training the model
trainer.train()
#Saving the model
trainer.save_model("Qlora finetuned")
model1 = AutoModelForCausalLM.from_pretrained("Qlora finetuned")
token = "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"
model1.push_to_hub("Amirkid/Qlora", use_auth_token = token)
