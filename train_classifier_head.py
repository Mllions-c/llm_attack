import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from stain.dataset_utils import load_dataset_custom

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset_name = "sst2"
num_epochs = 3
batch_size = 8
learning_rate = 2e-5

class Args:
    def __init__(self, dataset_name, num_examples):
        self.dataset_name = dataset_name
        self.num_examples = num_examples

args = Args(dataset_name=dataset_name, num_examples=500)

device = torch.device("cpu")
print(f"Using device: {device}")
if device.type == "mps":
    print(f"MPS memory allocated: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to {tokenizer.pad_token}")
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'
        print("Added and set tokenizer.pad_token to [PAD]")
model.eval()

num_labels = 4 if dataset_name == "AG-News" else 2
classifier_head = nn.Linear(model.config.hidden_size, num_labels).to(device)
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
        print(f"texts length: {len(texts)}, labels shape: {labels.shape if isinstance(labels, torch.Tensor) else len(labels)}")
        if isinstance(labels, torch.Tensor):
            if labels.shape[0] != batch_size:
                raise ValueError(f"labels shape {labels.shape} does not match batch_size {batch_size}")
        else:
            if len(labels) != batch_size:
                raise ValueError(f"labels length {len(labels)} does not match batch_size {batch_size}")
        labels = labels.to(device, dtype=torch.long) if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long).to(device)
        print(f"labels dtype: {labels.dtype}")
        print(f"labels values: {labels}")
        
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][:, -1, :].to(dtype=torch.float32)
        
        logits = classifier_head(hidden_states) 
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

save_path = f"classifier_head_{model_name.replace('/', '_')}_{dataset_name}.pt"
torch.save(classifier_head.state_dict(), save_path)
print(f"Classifier head saved to {save_path}")