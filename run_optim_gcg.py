import os
from stain.generate_and_save_category_ids import generate_and_save_category_ids, add_words_to_category_ids
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import numpy as np
import time
import random
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, AutoModelForMaskedLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer
from torch import Tensor
import torch.nn.functional as F
from stain.dataset_utils import load_dataset_custom
from stain.prompt_optim import compute_perplexity, compute_use_similarity, get_prediction_model, optimize_adversarial_suffix
from stain.generate_and_save_category_ids import load_category_ids, load_emotional_words, select_candidate_ids
from prediction_utils import get_prediction_call_model
from sentence_transformers import SentenceTransformer

use_model = SentenceTransformer('all-MiniLM-L6-v2')

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset_name = "sst2"
num_examples = 100
n_epochs = 100
batch_size = 10
top_k = 50
kl_every = 1
gamma = 0.0
similarity_threshold = 0.5 if dataset_name == "AG-News" else 0.5
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
        self.similarity_threshold = similarity_threshold
        self.ppl = ppl
        self.alpha = 1.0
        self.beta = 0.2
        self.gamma = 0.4

args = Args()

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'

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

classifier_head_path = f"./stain/head/classifier_head_{args.model_name.replace('/', '_')}_{args.dataset_name}.pt"
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
            f.write(f"Sample {result['sample']}: {result['original_prompt']} -> {result['optimized_prompt']} (Epoch {result['epoch']})\n")
            f.write(f"  Similarity: {result['similarity']:.4f}\n")
            f.write(f"  Perplexity: {result['perplexity']:.4f}\n")
    print(f"Results saved to {filepath}")
    print(f"Average Similarity: {avg_similarity:.4f}")
    print(f"Average Perplexity: {avg_perplexity:.4f}")
    print(f"Execution Time: {execution_time:.2f} seconds")

def select_candidate_ids_for_agnews(target_label, category_ids):
    target_words = []
    for category, ids in enumerate(category_ids):
        if category == target_label:
            target_words.extend([tid for tid in ids if len(bert_tokenizer.decode([tid]).strip()) > 2])
            break
    if not target_words:
        print(f"Warning: No valid candidate IDs for target_label {target_label}")
    return target_words

successful_attacks = 0
total_samples = len(dataset_class)
attack_results = []

print("Starting adversarial optimization...")
start_time = time.time()
category_ids, positive_ids, negative_ids = load_emotional_words(bert_tokenizer, dataset_name)

if args.dataset_name == "AG-News":
    category_ids = load_category_ids()

for i, (prompt_text, ground_truth_label) in enumerate(dataset_class):
    print(f"Processing sample {i+1}/{total_samples}")
    
    if args.dataset_name == "AG-News":
        possible_labels = [l for l in range(4) if l != ground_truth_label]
        label_priority = {
            0: [1, 2, 3],
            1: [0, 2, 3],
            2: [0, 1, 3],
            3: [2, 0, 1],
        }
        target_label = label_priority[ground_truth_label][0]
        candidate_ids = select_candidate_ids_for_agnews(target_label, category_ids)
    else:
        target_label = 0 if ground_truth_label == 1 else 1
        candidate_ids = select_candidate_ids(dataset_name, target_label, positive_ids, negative_ids)
        category_ids = None

    optimized_text = optimize_adversarial_suffix(
        model=model,
        tokenizer=tokenizer,
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer,
        classifier_model=classifier_model,
        classifier_tokenizer=classifier_tokenizer,
        orig_prompt=prompt_text,
        orig_pred=ground_truth_label,
        target_label=target_label,
        dataset_name=args.dataset_name,
        suffix_len=20 if args.dataset_name == "AG-News" else 8,
        similarity_threshold=args.similarity_threshold,
        n_steps=n_epochs,
        batch_size=batch_size,
        top_k=top_k,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        use_model=use_model,
        candidate_ids=candidate_ids,
        classifier_head=classifier_head
    )
    
    api_new_pred = get_prediction_call_model(model_name, optimized_text, dataset=args.dataset_name)    
    similarity = 0.0
    perplexity = 0.0

    similarity = compute_use_similarity(prompt_text, optimized_text, use_model)
    perplexity = compute_perplexity(bert_model, bert_tokenizer, optimized_text, device)

    
    if api_new_pred != ground_truth_label  and similarity >= args.similarity_threshold and perplexity <= 70:
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
        success_file = "successful_attacks.txt"
        with open(success_file, "a") as f:
            f.write(f"Original Prompt: {prompt_text} | Optimized Prompt: {optimized_text}\n")
    else:
        failure_reasons = []
        if api_new_pred == ground_truth_label :
            failure_reasons.append(f"Prediction unchanged (Original: {ground_truth_label}, New: {api_new_pred})")
        if similarity < args.similarity_threshold:
            failure_reasons.append(f"Similarity too low (Similarity: {similarity:.4f}, Threshold: {args.similarity_threshold})")
        print(f"Attack failed: {prompt_text} -> {optimized_text} (Similarity: {similarity:.4f})")
        print(f"Failure reasons: {', '.join(failure_reasons)}")

end_time = time.time()
execution_time = end_time - start_time
failed_attacks = total_samples - successful_attacks
asr = (successful_attacks / total_samples) * 100 if total_samples > 0 else 0
print(f"Optimization finished.")
save_attack_results(successful_attacks, failed_attacks, total_samples, asr, attack_results, execution_time, model_name)