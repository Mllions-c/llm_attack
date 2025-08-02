import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from evil_twins.dataset_utils import load_dataset_custom

# 参数设置
model_name = "EleutherAI/pythia-1.4b"
dataset_name = "AG-News"
num_epochs = 10
batch_size = 8
learning_rate = 1e-5 

class Args:
    def __init__(self, dataset_name, num_examples):
        self.dataset_name = dataset_name
        self.num_examples = num_examples

args = Args(dataset_name=dataset_name, num_examples=2000)

device = torch.device("cpu")
print(f"Using device: {device}")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device) 
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

model.eval()

num_labels = 4 if dataset_name == "AG-News" else 2
classifier_head = nn.Linear(model.config.hidden_size, num_labels).to(device)
classifier_head.weight.data.normal_(mean=0.0, std=0.02) 
classifier_head.bias.data.zero_()
optimizer = AdamW(classifier_head.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

dataset = load_dataset_custom(args)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Training classifier head for {dataset_name}...")
for epoch in range(num_epochs):
    classifier_head.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}")
        texts, labels = batch
        labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device, dtype=torch.long)
        print(f"Labels values: {labels}") 

        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][:, -1, :].float()
            print(f"Hidden states min/max: {hidden_states.min()}, {hidden_states.max()}") 
        
        hidden_states = torch.clamp(hidden_states, min=-1e9, max=1e9)
        
        logits = classifier_head(hidden_states)
        print(f"Logits min/max: {logits.min()}, {logits.max()}") 
        loss = loss_fn(logits, labels)
        print(f"Batch loss: {loss.item()}") 
        
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(classifier_head.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

save_path = f"classifier_head_{model_name.replace('/', '_')}_{dataset_name}.pt"
torch.save(classifier_head.state_dict(), save_path)
print(f"Classifier head saved to {save_path}")