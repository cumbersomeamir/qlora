#pip3 install -U bitandbytes datasets scikit-learn
#pip3 install -U git+https://github.com/huggingface/accelerate.git git+https://github.com/huggingface/peft.git git+https://github.com/huggingface/transformers.git 


#Importing the relevant Libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import transformers
from datasets import load_dataset


#Loading the model and defining the BitsAndBytes Config
model_id = "huggyllama/llama-65b"
bnb_config = BitsAndBytesConfig(load_in_4bit = True,
                                bnb_4bit_use_double_quant = True,
                                bnb_4bit_quant_type = "nf4",
                                bnb_4bit_compute_dtype = torch.bfloat16
                                )
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config, device_map = {"": 0})

#Enabling Gradient Checkpointing
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

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
                    lora_dropout = 0.05,
                    bias = "none",
                    task_type = "CAUSAL_LM"
                    )

#Converting model to peft model
model = get_peft_model(model, config)
#Calling the print trainable parameters function
print_trainable_parameters(model)

#Loading the Dataset
data = load_dataset("Amirkid/jokes")
data = data.map(lambda samples : tokenizer(samples["text"]), batched = True)


#Adding the pad token 
tokenizer.pad_token = tokenizer.eos_token

#Initialising the Trainer
trainer = transformers.Trainer(
    model = model,
    train_dataset = data["train"],
    args = transformers.TrainingArguments(
        per_device_train_batch_size =1,
        gradient_accumulation_steps= 4,
        warmup_steps = 2,
        max_steps = 50,
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 1,
        output_dir = "outputs",
        optim = "paged_adamw_8bit"
    ),
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm = False)
    )
#Use cache or not
model.config.use_cache = False
#Training the model
trainer.train()
#Saving the model
#trainer.save_model("Qlora finetuned")

#model1 = AutoModelForCausalLM.from_pretrained("Qlora finetuned")
model.save_pretrained("reddit-Qlora-llama65B")
token = "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"
model.push_to_hub("Amirkid/reddit-llama65B-Qlora", use_auth_token = token)
