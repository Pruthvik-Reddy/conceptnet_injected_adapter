import torch
from transformers import BertTokenizer, BertForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, TrainingArguments,AdapterTrainer,AdapterConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
corpus_file = "./conceptnet_data/conceptnet_corpus.txt"

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=corpus_file,
    block_size=128  
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15  
)

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
adapter_config = AdapterConfig.load("houlsby")

num_adapters = model.config.num_hidden_layers
adapter_names=[]
for i in range(num_adapters):
    adapter_name = f"adapter_{i}"
    adapter_names.append(adapter_name)
    model.add_adapter(adapter_name,config=adapter_config)
    break
    
model.set_active_adapters(adapter_names)
model.train_adapter(adapter_names)

training_args = TrainingArguments(
    output_dir="./adapter_conceptnet_model_3", 
    num_train_epochs=3, 
    per_device_train_batch_size=16,  
    save_steps=500,
    learning_rate=5e-5,  
    warmup_steps=500,  
    logging_dir="./logs3", 
    logging_steps=100, 
    #evaluation_strategy="steps",  
    #eval_steps=500, 
    dataloader_num_workers=4 
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
model.save_pretrained("./adapter_conceptnet_model_3")
