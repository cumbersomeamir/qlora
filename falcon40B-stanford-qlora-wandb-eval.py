#pip3 install -U wandb einops bitsandbytes datasets scikit-learn git+https://github.com/huggingface/accelerate.git git+https://github.com/huggingface/peft.git git+https://github.com/huggingface/transformers.git 


#Importing the relevant Libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import transformers
from datasets import load_dataset
import wandb
import os


os.environ["WANDB_PROJECT"] = "exprmt"
os.environ["WANDB_LOG_MODEL"] = "true"

#Loading the model and defining the BitsAndBytes Config
model_id = "tiiuae/falcon-40b"
bnb_config = BitsAndBytesConfig(load_in_4bit = True,
                                bnb_4bit_use_double_quant = True,
                                bnb_4bit_quant_type = "nf4",
                                bnb_4bit_compute_dtype = torch.bfloat16
                                )
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config, device_map = {"": 0}, trust_remote_code=True)

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
                    target_modules = ['query_key_value'],
                    lora_dropout = 0.05,
                    bias = "none",
                    task_type = "CAUSAL_LM"
                    )

#Converting model to peft model
model = get_peft_model(model, config)
#Calling the print trainable parameters function
print_trainable_parameters(model)
#Adding the pad token 
tokenizer.pad_token = tokenizer.eos_token

#Loading the Dataset
data = load_dataset("Amirkid/stanford_alpaca_new")
data = data.map(lambda samples : tokenizer(samples["text"]), batched = True)

train_data = data["train"].train_test_split(test_size=0.2)['train']
valid_data = data["train"].train_test_split(test_size=0.2)['test']




#Initialising the Trainer
trainer = transformers.Trainer(
    model = model,
    train_dataset = data["train"],
    args = transformers.TrainingArguments(
        per_device_train_batch_size =16,
        gradient_accumulation_steps= 4,
        warmup_steps = 50,
        max_steps = 150,
        learning_rate = 1e-4,
        fp16 = True,
        logging_steps = 1,
        output_dir = "outputs",
        optim = "paged_adamw_8bit",
        report_to = 'wandb',
        evaluation_strategy = "steps",
        eval_steps = 50
    ),
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm = False)
    )
#Use cache or not
model.config.use_cache = False
#Training the model
trainer.train()
#Saving the model
#trainer.save_model("Qlora finetuned")
username = "Amirkid"
model_name = "stanford_alpaca-falcon40B-Qlora"

#model1 = AutoModelForCausalLM.from_pretrained("Qlora finetuned")
model.save_pretrained(f"{username}/{model_name}")
token = "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"
model.push_to_hub("Amirkid/stanford-falcon40B-qlora-200steps-new", use_auth_token = token)
