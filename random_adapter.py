import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, TrainingArguments,AdapterTrainer,AdapterConfig

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
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

model.save_pretrained("./random_adapter")
