import torch
from transformers import BertTokenizer, BertForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, TrainingArguments,AdapterTrainer,AdapterConfig
from torchsummary import summary
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
corpus_file = "./conceptnet_data/conceptnet_corpus_2.txt"

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


print(model)

for w in model.named_parameters():
  # if w[1].requires_grad == True:
  print(w[0], end=",\t")
  print(w[1].shape)
summary(model,input_size=(768,),depth=1,batch_dim=1, dtypes=['torch.IntTensor']) 



print(" Calculating BERT, Adapter, Active and Total Parameters :")
num_params = sum(p.numel() for p in model.parameters())

print("Number of parameters in BERT: ", num_params)
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
num_trainable_params = sum(p.numel() for p in trainable_params)
print("\nNumber of Trainable Parameters:", num_trainable_params)