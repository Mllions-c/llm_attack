import os  
from evil_twins import generate_and_save_category_ids
from evil_twins.generate_and_save_category_ids import generate_and_save_category_ids
import torch 
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
import math  
import numpy as np 
import time 
import random  
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, AutoModelForMaskedLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer  # 从 transformers 库导入模型和分词器
from torch import Tensor  
import torch.nn.functional as F 
from evil_twins.dataset_utils import load_dataset_custom 
from evil_twins.prompt_optim import optimize_adversarial_suffix 
from evil_twins.generate_and_save_category_ids import load_category_ids,load_emotional_words,select_candidate_ids
from prediction_utils import get_prediction_call_model
from sentence_transformers import SentenceTransformer  

use_model = SentenceTransformer('all-MiniLM-L6-v2') 

# 定义参数（硬编码）
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset_name = "StrategyQA" 
num_examples = 100 
n_epochs = 100  
batch_size = 10 
top_k = 50  
kl_every = 1  
gamma = 0.0  
suffix_mode = False 
similarity_threshold = 0.4 if dataset_name == "AG-News" else 0.5 
ppl = 0.7 

class Args:  
    def __init__(self):
        self.model_name = model_name  
        self.dataset_name = dataset_name 
        self.num_examples = num_examples  
        self.n_epochs = n_epochs 
        self.batch_size = batch_size
        self.top_k = top_k  
        self.kl_every = kl_every 
        self.gamma = gamma 
        self.suffix_mode = suffix_mode 
        self.similarity_threshold = similarity_threshold 
        self.ppl = ppl 
        
        self.beta = 0.5  
        self.gamma = 0.1 

args = Args() 


model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name) 
bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased") 
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to {tokenizer.pad_token}")
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'
        print("Added and set tokenizer.pad_token to [PAD]")
if args.dataset_name == "AG-News":  
    classifier_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news", num_labels=4) 
    classifier_tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news") 
else: 
   
    classifier_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2", num_labels=2) 
    classifier_tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device) 
bert_model.to(device)
classifier_model.to(device) 
use_model.to(device) 

classifier_head_path = f"classifier_head_{args.model_name.replace('/', '_')}_{args.dataset_name}.pt"
num_labels = 4 if args.dataset_name == "AG-News" else 2 
classifier_head = torch.nn.Linear(model.config.hidden_size, num_labels).to(device)
if os.path.exists(classifier_head_path):
    classifier_head.load_state_dict(torch.load(classifier_head_path))
    print(f"Loaded classifier head from {classifier_head_path}")
else:
    print(f"Classifier head not found at {classifier_head_path}. Please train the classifier head first.")
    raise FileNotFoundError(f"Classifier head not found at {classifier_head_path}")
classifier_head.eval()
dataset_class = load_dataset_custom(args) 


def get_prediction_model(model, tokenizer, classifier_head, text: str) -> int:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, -1, :] 
        logits = classifier_head(hidden_states)  
        pred = torch.argmax(logits, dim=-1).item()
    return pred

def save_attack_results(successful_attacks, failed_attacks, total_samples, asr, attack_results, execution_time, model_name, filepath_prefix="attack_results"):
    filepath = f"{filepath_prefix}_{model_name.replace('/', '_')}_{args.dataset_name}.txt"
    avg_similarity = sum(result['similarity'] for result in attack_results) / len(attack_results) if attack_results else 0.0 
    avg_perplexity = sum(result['perplexity'] for result in attack_results) / len(attack_results) if attack_results else 0.0 
    with open(filepath, "w") as f:  
        f.write(f"Successful Attacks: {successful_attacks}\n") 
        f.write(f"Failed Attacks: {failed_attacks}\n") 
        f.write(f"Total Samples: {total_samples}\n") 
        f.write(f"Attack Success Rate: {asr:.2f}%\n") 
        f.write(f"Execution Time: {execution_time:.2f} seconds\n") 
        f.write("\nSuccessful Attack Details:\n") 
        f.write(f"Average Similarity: {avg_similarity:.4f}\n")
        f.write(f"Average Perplexity: {avg_perplexity:.4f}\n") 
        for result in attack_results: 
            f.write(f"Sample {result['sample']}: {result['original_prompt']} -> {result['optimized_prompt']} (Epoch {result['epoch']})\n")  # 写入样本编号、原始提示、优化提示和轮数
            f.write(f"  Similarity: {result['similarity']:.4f}\n") 
            f.write(f"  Perplexity: {result['perplexity']:.4f}\n")  
    print(f"Results saved to {filepath}") 
    print(f"Average Similarity: {avg_similarity:.4f}") 
    print(f"Average Perplexity: {avg_perplexity:.4f}") 
    print(f"Execution Time: {execution_time:.2f} seconds") 

"""
功能概述：
compute_use_similarity 方法计算两个文本之间的语义相似度，使用 SentenceTransformer 模型。
它将文本编码为嵌入向量，计算余弦相似度，返回 0-1 范围内的值。
用于评估原始提示和优化提示的语义一致性。
"""
def compute_use_similarity(text1: str, text2: str, use_model) -> float:
    embedding1 = use_model.encode(text1, convert_to_tensor=True) 
    embedding2 = use_model.encode(text2, convert_to_tensor=True) 
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0).item() 
    return similarity  # 返回相似度值

def select_candidate_ids_for_agnews(target_label, category_ids):
    target_words = [] 
    for category, ids in enumerate(category_ids): 
        if category == target_label: 
            target_words.extend(ids)  
            break 
    return target_words 

# 主循环
successful_attacks = 0 
total_samples = len(dataset_class) 
attack_results = [] 

print("Starting adversarial optimization...") 
start_time = time.time() 

if args.dataset_name == "AG-News":
    category_ids = load_category_ids() 
else:
    category_ids = None 

for i, (prompt_text, label) in enumerate(dataset_class):
    print(f"Processing sample {i+1}/{total_samples}")  
    
    
    orig_pred = get_prediction_model(model, tokenizer, classifier_head, prompt_text)
    api_old_pred = get_prediction_call_model(model_name, prompt_text, dataset=args.dataset_name) 
    if args.dataset_name == "AG-News":  
        possible_labels = [l for l in range(4) if l != orig_pred] 
        label_priority = { 
            0: [1, 2, 3],
            1: [0, 2, 3], 
            2: [0, 1, 3], 
            3: [2, 0, 1], 
        }
        target_label = label_priority[orig_pred][0] 
        category_ids = load_category_ids()  # 加载与AG-News数据集类别相关的词汇ID
        candidate_ids = select_candidate_ids_for_agnews(target_label, category_ids)

    else: 
        target_label = 0 if orig_pred == 1 else 1 
        category_ids, positive_ids, negative_ids = load_emotional_words(bert_tokenizer, dataset_name)  # 加载情感词汇（正面和负面词），category_ids通常为空列表
        candidate_ids = select_candidate_ids(dataset_name, target_label, category_ids, positive_ids, negative_ids)  # 根据数据集和目标类别选择候选词ID（如正面情感词）
        candidate_ids=None

    optimized_text = optimize_adversarial_suffix( 
        model=model,
        tokenizer=tokenizer,
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer,
        classifier_model=classifier_model,
        classifier_tokenizer=classifier_tokenizer,
        orig_prompt=prompt_text, 
        orig_pred = orig_pred,
        target_label=target_label, 
        dataset_name=args.dataset_name,  
        suffix_len=25 if args.dataset_name == "AG-News" else 15,
        similarity_threshold=args.similarity_threshold,
        n_steps=n_epochs, 
        batch_size=batch_size, 
        top_k=top_k, 
        beta=args.beta,   
        gamma=args.gamma, 
        use_model=use_model,
        candidate_ids=candidate_ids,
        classifier_head=classifier_head  # 传递分类头

    )
    
    new_pred = get_prediction_model(model, tokenizer, classifier_head, optimized_text)
    
    similarity = 0.0 
    perplexity = 0.0  
    

    similarity = compute_use_similarity(prompt_text, optimized_text, use_model) 

    bert_inputs = bert_tokenizer(optimized_text, return_tensors="pt", truncation=True, padding=True, max_length=512) 
    bert_inputs = {k: v.to(bert_model.device) for k, v in bert_inputs.items()}  
    with torch.no_grad(): 
        bert_outputs = bert_model(**bert_inputs, labels=bert_inputs["input_ids"])
        loss = bert_outputs.loss 
        perplexity = math.exp(loss.item()) 

    if orig_pred != new_pred and similarity >= args.similarity_threshold and perplexity <= 70:
        successful_attacks += 1 
        print(f"Attack succeeded: {prompt_text} -> {optimized_text} (Similarity: {similarity:.4f})") 
        attack_results.append({  
            "sample": i+1,
            "original_prompt": prompt_text,
            "optimized_prompt": optimized_text,
            "epoch": n_epochs,
            "similarity": similarity,
            "perplexity": perplexity
        })
    else: 
        failure_reasons = []
        if orig_pred == new_pred: 
            failure_reasons.append(f"Prediction unchanged (Original: {orig_pred}, New: {new_pred})")
        if similarity < args.similarity_threshold: 
            failure_reasons.append(f"Similarity too low (Similarity: {similarity:.4f}, Threshold: {args.similarity_threshold})")
        print(f"Attack failed: {prompt_text} -> {optimized_text} (Similarity: {similarity:.4f})") 
        print(f"Failure reasons: {', '.join(failure_reasons)}") 

end_time = time.time()  
execution_time = end_time - start_time  
failed_attacks = total_samples - successful_attacks  
asr = (successful_attacks / total_samples) * 100 if total_samples > 0 else 0 
print(f"Optimization finished.") 
save_attack_results(successful_attacks, failed_attacks, total_samples, asr, attack_results, execution_time, model_name)  # 保存攻击结果