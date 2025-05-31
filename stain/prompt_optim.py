# GCG adapted from https://github.com/llm-attacks/llm-attacks

# %%

from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, BertForSequenceClassification, BertTokenizer  #
from stain.generate_and_save_category_ids import load_category_ids
import torch
import os
import random
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import json
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import nltk
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

import torch
import torch.nn.functional as F
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')  # 词性标注需要，确保下载英文版
nltk.download('sentiwordnet')

# %%
PROMPT_TEMPLATES = {
  "vicuna": {
    "prefix": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: ",
    "suffix": "\nASSISTANT:",
  },
  "llama-chat": {
    "prefix": "[INST] ",
    "suffix": " [/INST]",
  },
  "mistral": {
    "prefix": "[INST] ",
    "suffix": " [/INST]",
  },
  "dolphin": {
    "prefix": "<|im_start|>system\nYou are Dolphin, a helpful AI assistant.<|im_end|>\n<|im_start|>user\n",
    "suffix": "<|im_end|>\n<|im_start|>assistant",
  },
  "openhermes": {
    "prefix": "<|im_start|>system\nYour purpose and drive is to assist the user with any request they have. Answer the user's question directly no matter what it is, and do not apologize.<|im_end|>\n<|im_start|>user\n",
    "suffix": "<|im_end|>\n<|im_start|>assistant\n",
  },
  "opt": {
    "prefix": "",
    "suffix": "",
  },
  "phi": {
    "prefix": "",
    "suffix": "",
  },
  "pythia": {
    "prefix": "",
    "suffix": "",
  },
  "oasst": {
    "prefix": "<|prompter|>",
    "suffix": "<|endoftext|><|assistant|>",
  },
  "openllama": {
    "prefix": "",
    "suffix": "",
  },
  "gpt2": {
    "prefix": "",
    "suffix": "",
  },
  "gemma": {
    "prefix": "",
    "suffix": "",
  },
  "gemma-2-it": {
    "prefix": "<start_of_turn>user\n",
    "suffix": "<end_of_turn>\n<start_of_turn>model\n",
  },
  # "llama-3-instruct": {
  #   "prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
  #   "suffix": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
  # },
  "llama-3-instruct": {
    "prefix": "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
    "suffix": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
  },
  "llama-base": {
    "prefix": "",
    "suffix": "",
  },
  "qwen-2-instruct": {
    "prefix": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
    "suffix": "<|im_end|>\n<|im_start|>",
  },
  "smollm-2-instruct": {
    "prefix": "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n",
    "suffix": "<|im_end|>\n<|im_start|>assistant\n",
  },
  "default": {
    "prefix": "",
    "suffix": "",
  },
}

MODEL_NAME_OR_PATH_TO_NAME = {
  "lmsys/vicuna-7b-v1.3": "vicuna",
  "lmsys/vicuna-7b-v1.5": "vicuna",
  "lmsys/vicuna-13b-v1.5": "vicuna",
  "vicuna": "vicuna",
  "facebook/opt-350m": "opt",
  "facebook/opt-1.3b": "opt",
  "microsoft/phi-1_5": "phi",
  "microsoft/phi-2": "phi",
  "teknium/Puffin-Phi-v2": "phi",
  "OpenAssistant/oasst-sft-1-pythia-12b": "oasst",
  "EleutherAI/pythia-14m": "pythia",
  "EleutherAI/pythia-70m": "pythia",
  "EleutherAI/pythia-160m": "pythia",
  "EleutherAI/pythia-410m": "pythia",
  "EleutherAI/pythia-1b": "pythia",
  "EleutherAI/pythia-1.4b": "pythia",
  "EleutherAI/pythia-2.8b": "pythia",
  "EleutherAI/pythia-6.9b": "pythia",
  "EleutherAI/pythia-12b": "pythia",
  "pythia": "pythia",
  "openlm-research/open_llama_3b_v2": "openllama",
  "mistralai/Mistral-7B-Instruct-v0.2": "mistral",
  "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
  "teknium/OpenHermes-2.5-Mistral-7B": "openhermes",
  "cognitivecomputations/dolphin-2.2.1-mistral-7b": "dolphin",
  "gpt2": "gpt2",
  "teknium/OpenHermes-13B": "openhermes",
  "meta-llama/Llama-2-7b-chat-hf": "llama-chat",
  "meta-llama/Llama-2-13b-chat-hf": "llama-chat",
  "google/gemma-2b-it": "gemma",
  "google/gemma-1.1-2b-it": "gemma",
  "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3-instruct",
  "meta-llama/Llama-3.1-8B-Instruct": "llama-3-instruct",
  "meta-llama/Llama-3.2-3B-Instruct": "llama-3-instruct",
  "meta-llama/Llama-3.2-1B-Instruct": "llama-3-instruct",
  "google/gemma-2-2b-it": "gemma-2-it",
  "google/gemma-2-9b-it": "gemma-2-it",
  "HuggingFaceTB/SmolLM2-135M-Instruct": "smollm-2-instruct",
  "HuggingFaceTB/SmolLM2-360M-Instruct": "smollm-2-instruct",
  "HuggingFaceTB/SmolLM2-1.7B-Instruct": "smollm-2-instruct",
  "Qwen/Qwen2.5-0.5B-Instruct": "qwen-2-instruct",
  "Qwen/Qwen2.5-1.5B-Instruct": "qwen-2-instruct",
  "Qwen/Qwen2.5-7B-Instruct": "qwen-2-instruct",
  "default": "default",
}


"""
功能概述：
attack_for_agnews 是 STAIN 系统中针对 AG-News 数据集的对抗性攻击方法，
方法使用 BERT 模型计算困惑度，使用模型计算损失，通过梯度近似选择最佳词，迭代优化直到预测翻转。
最终返回修改后的对抗性文本，同时控制语言自然性（通过困惑度阈值）。
"""
def attack_for_agnews(
    bert_model,  
    classifier_model,
    classifier_tokenizer,
    bert_tokenizer, 
    input_text: str, 
    target_label: int, 
    device, 
    max_step: int = 15, 
    perplexity_threshold: float = 8.0, 
    alpha: float = 1.0,  # 总损失中对抗性损失的权重
    beta: float = 0.5,   # 总损失中相似度损失的权重
    gamma: float = 0.1,  # 总损失中困惑度损失的权重
    use_model=None,  
    model=None,  
    tokenizer=None,
    candidate_ids=None
):
    classifier_model.eval() 
    bert_model.eval()  

    
    inputs = bert_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)  
    input_ids = inputs["input_ids"].to(device) 
    attention_mask = inputs["attention_mask"].to(device)
    
    connector_file = "connector_structures.json"
    if not os.path.exists(connector_file): 
        raise FileNotFoundError(f"Connector structures file not found at {connector_file}.") 
    with open(connector_file, "r") as f: 
        connector_structures = json.load(f)
    
     # 使用SentenceTransformer选择与输入文本最匹配的连接短语
    prompt_embedding = use_model.encode(input_text, convert_to_tensor=True) 
    connector_phrases = [phrase for phrase, _ in connector_structures] 
    connector_embeddings = use_model.encode(connector_phrases, convert_to_tensor=True)
    similarities = torch.nn.functional.cosine_similarity(prompt_embedding, connector_embeddings, dim=-1) 

    
    structure_idx = torch.argmax(similarities).item() 
    print(f"Selected connector phrase: '{connector_phrases[structure_idx]}' with similarity {similarities[structure_idx]:.4f}")

    structure_name, structure_tokens = connector_structures[structure_idx]
    structure_token_ids = [bert_tokenizer.encode(word, add_special_tokens=False)[0] for word in structure_tokens] 

    seen_tokens = set(structure_token_ids)  
    insert_ids = structure_token_ids.copy() 

    # 初始化
    num_candidates = random.randint(1, 3) 
    available_tokens = [tid for tid in candidate_ids if tid not in seen_tokens] 
    for _ in range(min(num_candidates, len(available_tokens))): 
        candidate_idx = torch.randint(0, len(available_tokens), (1,)).item()
        insert_ids.append(available_tokens[candidate_idx]) 
        seen_tokens.add(available_tokens[candidate_idx]) 
        available_tokens.pop(candidate_idx)

    
    orig_ids_list = input_ids[0].tolist() 
    adv_input_ids = orig_ids_list[:-1] + insert_ids + [orig_ids_list[-1]] 
    adv_input_ids = torch.tensor(adv_input_ids, device=device).unsqueeze(0)  
    attention_mask = torch.ones_like(adv_input_ids).to(device)  # 

    adv_text = bert_tokenizer.decode(adv_input_ids[0], skip_special_tokens=True)  
    inputs = classifier_tokenizer(adv_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad(): 
        outputs = classifier_model(**inputs) 
        orig_logits = outputs.logits 
        orig_probs = F.softmax(orig_logits, dim=-1) 
        orig_pred = torch.argmax(orig_probs, dim=-1).item()

    if orig_pred == target_label:
        print("Attack already successful after phrase insertion.")
        return adv_text 

    
    target_tensor = torch.tensor([target_label], device=device)  
    step = 0  
    while step < max_step: 
        
        current_text = bert_tokenizer.decode(adv_input_ids[0], skip_special_tokens=True) 
        bert_inputs = {
            "input_ids": adv_input_ids, 
            "attention_mask": attention_mask
        }
        with torch.no_grad(): 
            bert_outputs = bert_model(**bert_inputs, labels=adv_input_ids)
            ppl_loss = bert_outputs.loss  # 获取困惑度损失
            perplexity = math.exp(ppl_loss.item()) 
            print(f"step {step+1}: Perplexity = {perplexity:.4f}") 

        # 计算文本相似度损失
        sim_loss = 1.0 - compute_use_similarity(input_text, current_text, use_model)

        embeddings = bert_model.get_input_embeddings()(adv_input_ids)
        embeddings.retain_grad()

        
        outputs = classifier_model(inputs_embeds=embeddings, attention_mask=attention_mask) 
        logits = outputs.logits 
        adv_loss = F.cross_entropy(logits, target_tensor)  # 计算对抗性损失（交叉熵损失）

        # 总损失 = 对抗性损失 + 相似度损失 + 困惑度损失
        total_loss = alpha * adv_loss + beta * sim_loss + gamma * ppl_loss 
        total_loss.backward() 

        grad = embeddings.grad.detach()

        embedding_matrix = bert_model.get_input_embeddings().weight.detach()

        best_loss = total_loss.item()
        best_pos = -1
        best_token = None

        
        for pos in range(1, adv_input_ids.size(1) - 1): 
            if attention_mask[0, pos] == 0:
                continue

            orig_token = adv_input_ids[0, pos].item() 
            orig_embedding = embedding_matrix[orig_token] 
            grad_at_pos = grad[0, pos]

            for cand_id in candidate_ids: 
                if cand_id == orig_token: 
                    continue
                cand_embedding = embedding_matrix[cand_id]
                delta_embedding = cand_embedding - orig_embedding 
                loss_change = torch.dot(grad_at_pos, delta_embedding)
                new_loss = total_loss.item() + loss_change.item()
                if new_loss < best_loss: 
                    best_loss = new_loss
                    best_pos = pos 
                    best_token = cand_id

        
        if best_pos == -1: 
            print("No better replacement found, stopping.")
            break

        adv_input_ids[0, best_pos] = best_token  
        step += 1

        with torch.no_grad():
            current_text = bert_tokenizer.decode(adv_input_ids[0], skip_special_tokens=True) 
            classifier_inputs = classifier_tokenizer(current_text, return_tensors="pt", truncation=True, padding=True, max_length=512) 
            classifier_inputs = {k: v.to(device) for k, v in classifier_inputs.items()} 
            logits = classifier_model(**classifier_inputs).logits 
            new_pred = logits.argmax(dim=-1).item() 
            print(f"step {step}: Prediction = {new_pred}")
            if new_pred != orig_pred: 
                print("Prediction flipped successfully!") 
                break

        
        if random.random() < 0.2:  
            suffix_start = len(orig_ids_list[:-1]) + len(structure_token_ids) 
            suffix_end = len(orig_ids_list[:-1]) + len(insert_ids) 
            valid_positions = list(range(suffix_start, suffix_end)) 

            if valid_positions:
                replace_pos = random.choice(valid_positions) 
                current_token = adv_input_ids[0, replace_pos].item() 
                current_word = bert_tokenizer.decode([current_token], skip_special_tokens=True) 

                
                pos_tags = nltk.pos_tag([current_word])
                current_pos = pos_tags[0][1]
                
                if current_pos.startswith('JJ'):  # 如果是形容词
                    target_pos = 'JJ'  # 设置目标词性为形容词
                elif current_pos.startswith('NN'):  # 如果是名词
                    target_pos = 'NN'  # 设置目标词性为名词
                elif current_pos.startswith('VB'):  # 如果是动词
                    target_pos = 'VB'  # 设置目标词性为动词
                else:
                    target_pos = None  

                if target_pos: 
                    
                    available_tokens = [] 
                    for tid in candidate_ids: 
                        if tid in seen_tokens or tid == current_token:  
                            continue
                        word = bert_tokenizer.decode([tid], skip_special_tokens=True) 
                        cand_pos_tags = nltk.pos_tag([word]) 
                        cand_pos = cand_pos_tags[0][1] 
                        if cand_pos.startswith(target_pos): 
                            available_tokens.append(tid)

                    if available_tokens:
                        new_token = random.choice(available_tokens)
                        adv_input_ids[0, replace_pos] = new_token 
                        seen_tokens.add(new_token) 
                        print(f"Randomly replaced token at position {replace_pos} (POS: {target_pos}): '{current_word}' -> '{bert_tokenizer.decode([new_token])}'") 
                    else:
                        print(f"No candidates with POS {target_pos} available for replacement at position {replace_pos}.") 
                else:
                    print(f"Skipping replacement at position {replace_pos}: unsupported POS {current_pos} for '{current_word}'.")

    
    adv_text = bert_tokenizer.decode(adv_input_ids[0], skip_special_tokens=True)
    return adv_text 




def load_emotional_words(bert_tokenizer, dataset_name, threshold=0.75, max_words=100):
   
    def get_words_by_category(category="positive"): 
        words = [] 
        for synset in wn.all_synsets(wn.ADJ):  
            word = synset.name().split('.')[0]
            senti_synsets = list(swn.senti_synsets(word, 'a'))
            if senti_synsets:
                s = senti_synsets[0] 
                if category == "positive" and s.pos_score() > s.neg_score() and s.pos_score() >= threshold: 
                    words.append(word) 
                elif category == "negative" and s.neg_score() > s.pos_score() and s.neg_score() >= threshold:  # 如果类别为“negative”，且负面得分大于正面得分且达到阈值
                    words.append(word) 
            if len(words) >= max_words: 
                break 
        return words

    negative_words = get_words_by_category(category="negative") 
    positive_words = get_words_by_category(category="positive")
    negative_ids = [bert_tokenizer.encode(word, add_special_tokens=False)[0] for word in negative_words if bert_tokenizer.encode(word, add_special_tokens=False)]  # 将负面词编码为词元 ID
    positive_ids = [bert_tokenizer.encode(word, add_special_tokens=False)[0] for word in positive_words if bert_tokenizer.encode(word, add_special_tokens=False)]  # 将正面词编码为词元 ID
    negative_ids = list(set(negative_ids)) 
    positive_ids = list(set(positive_ids)) 
    return [], positive_ids, negative_ids

def compute_use_similarity(prompt, optimized_prompt, use_model):
    prompt_embedding = use_model.encode(prompt, convert_to_tensor=True)  #
    optimized_embedding = use_model.encode(optimized_prompt, convert_to_tensor=True)
    similarity = torch.nn.functional.cosine_similarity(prompt_embedding, optimized_embedding, dim=0).item()
    return similarity  # 返回语义相似度值


def select_candidate_ids(dataset_name, target_label, category_ids, positive_ids, negative_ids):
    return positive_ids if target_label == 1 else negative_ids  # 目标为正面（1）选择正面词，否则选择负面词


"""
功能概述：
generate_suffix 是 STAIN核心方法  通过迭代优化后缀 S = C ⊕ W（C 为连接短语，W 为对抗性词序列），使模型预测翻转，同时控制语言自然性。
结合分类器损失和 top-k 采样选择对抗性词

"""
def generate_suffix(
    bert_model,
    classifier_model, 
    classifier_tokenizer, 
    bert_tokenizer,  
    input_ids, 
    orig_len, 
    suffix_len, 
    structure_token_ids,
    candidate_ids,  
    additional_connector_ids,
    target_tensor,
    n_steps,
    top_k, 
    device, 
    dataset_name, 
    alpha: float = 1.0,  # 总损失中对抗性损失的权重
    beta: float = 0.5,   # 总损失中相似度损失的权重
    gamma: float = 0.1,  # 总损失中困惑度损失的权重
    use_model=None, 
    model=None, 
    tokenizer=None
):
    seen_tokens = set(structure_token_ids) 
    emotional_words_count = 0
    previous_loss = float('inf') 
    loss_stagnation_count = 0 
    loss_threshold = 1e-5 
    max_stagnation_steps = 5  
    max_emotional_words = random.randint(1, 3) 
    perplexity_threshold = 70

    
    orig_prompt = bert_tokenizer.decode(input_ids[0, :orig_len], skip_special_tokens=True) 

    for step in range(n_steps):
        temperature = max(0.4, 0.7 - (step / n_steps) * 0.3) 
        current_text = bert_tokenizer.decode(input_ids[0], skip_special_tokens=True) 

        # 计算分类器损失
        inputs = classifier_tokenizer(current_text, return_tensors="pt", truncation=True, padding=True, max_length=512) 
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = classifier_model(**inputs) 
        classifier_logits = outputs.logits 
        adv_loss = F.cross_entropy(classifier_logits, target_tensor)  # 计算对抗性损失（交叉熵损失）

        # 计算困惑度损失
        bert_inputs = bert_tokenizer(current_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        bert_inputs = {k: v.to(bert_model.device) for k, v in bert_inputs.items()}  

        with torch.no_grad(): 
            bert_outputs = bert_model(**bert_inputs, labels=bert_inputs["input_ids"]) 
            ppl_loss = bert_outputs.loss  # 获取困惑度损失

        # 计算相似度损失  # 计算相似度损失（1 - 原始提示与当前文本的相似度）
        sim_loss = 1.0 - compute_use_similarity(orig_prompt, current_text, use_model) 

        #  总损失 = 对抗性损失 + 相似度损失 + 困惑度损失
        total_loss = alpha * adv_loss + beta * sim_loss + gamma * ppl_loss 
        current_loss = total_loss.item()
        print(f"Step {step+1}: Loss = {current_loss}") 

        # 早停判断
        if step > 0 and abs(current_loss - previous_loss) < loss_threshold: 
            loss_stagnation_count += 1 
            if loss_stagnation_count >= max_stagnation_steps: 
                print(f"Early stopping at Step {step+1}: Loss stagnated for {max_stagnation_steps} steps.")  # 打印日志，记录早停
                break
        else:
            loss_stagnation_count = 0 
        previous_loss = current_loss

        total_loss.backward() 

        with torch.no_grad():
            for pos in range(len(structure_token_ids), suffix_len):
                connector_prob = 0.3 
                secondary_connector_prob = 0.1
                if pos == len(structure_token_ids) and random.random() < connector_prob:  
                    connector_idx = torch.randint(0, len(additional_connector_ids), (1,)).item() 
                    input_ids[0, orig_len + pos] = additional_connector_ids[connector_idx]
                    seen_tokens.add(additional_connector_ids[connector_idx])
                    continue
                if pos > len(structure_token_ids) and random.random() < secondary_connector_prob: 
                    connector_idx = torch.randint(0, len(additional_connector_ids), (1,)).item() 
                    input_ids[0, orig_len + pos] = additional_connector_ids[connector_idx]
                    seen_tokens.add(additional_connector_ids[connector_idx]) 
                    continue

                if emotional_words_count >= max_emotional_words: 
                    break 

                total_loss_value = total_loss.item()


                new_token = generate_next_token( 
                    bert_model=bert_model,
                    input_ids=input_ids, 
                    position=orig_len + pos,
                    top_k=top_k, 
                    seen_tokens=seen_tokens,
                    bert_tokenizer=bert_tokenizer, 
                    candidate_ids=candidate_ids,
                    temperature=temperature,
                    classifier_model=classifier_model, 
                    classifier_tokenizer=classifier_tokenizer,
                    target_tensor=target_tensor,
                    total_loss_value=total_loss_value
                )
                if new_token is None: 
                    available_tokens = [tid for tid in candidate_ids if tid not in seen_tokens] 
                    if not available_tokens: 
                        break 
                    new_token = available_tokens[torch.randint(0, len(available_tokens), (1,)).item()]
                input_ids[0, orig_len + pos] = new_token 
                seen_tokens.add(new_token) 
                emotional_words_count += 1

        classifier_model.zero_grad()

    # 以20%概率随机替换后缀中的一个词，仅在后缀达到最大长度且无填充词元时执行
    if random.random() < 0.2 and bert_tokenizer.pad_token_id not in input_ids[0, orig_len:orig_len+suffix_len]:  # 以20%概率执行，且后缀中无PAD词元
        
        suffix_start = orig_len + len(structure_token_ids)
        suffix_end = orig_len + suffix_len 
        valid_positions = [pos for pos in range(suffix_start, suffix_end) if input_ids[0, pos] != bert_tokenizer.pad_token_id] 

        if valid_positions:  # 如果存在可替换的位置
            replace_pos = random.choice(valid_positions)  # 随机选择一个替换位置
            current_token = input_ids[0, replace_pos].item() 
            current_word = bert_tokenizer.decode([current_token], skip_special_tokens=True) 

            
            pos_tags = nltk.pos_tag([current_word])  
            current_pos = pos_tags[0][1] 
            
            if current_pos.startswith('JJ'):
                target_pos = 'JJ'  
            elif current_pos.startswith('NN'): 
                target_pos = 'NN' 
            elif current_pos.startswith('VB'): 
                target_pos = 'VB' 
            else:
                target_pos = None

            if target_pos: 
                available_tokens = [] 
                for tid in candidate_ids: 
                    if tid in seen_tokens or tid == current_token:
                        continue
                    word = bert_tokenizer.decode([tid], skip_special_tokens=True) 
                    cand_pos_tags = nltk.pos_tag([word]) 
                    cand_pos = cand_pos_tags[0][1] 
                    if cand_pos.startswith(target_pos):
                        available_tokens.append(tid) 

                if available_tokens:
                    new_token = random.choice(available_tokens)
                    input_ids[0, replace_pos] = new_token 
                    seen_tokens.add(new_token) 
                    print(f"Randomly replaced token at position {replace_pos} (POS: {target_pos}): '{current_word}' -> '{bert_tokenizer.decode([new_token])}'")
                else:
                    print(f"No candidates with POS {target_pos} available for replacement at position {replace_pos}.")
            else:
                print(f"Skipping replacement at position {replace_pos}: unsupported POS {current_pos} for '{current_word}'.")

    return input_ids, seen_tokens 


def append_strong_suffix(
    input_ids, 
    orig_len,  
    suffix_len, 
    bert_tokenizer,  
    additional_connector_ids, 
    intensifier_ids, 
    candidate_ids, 
    seen_tokens, 
    dataset_name
):
    with torch.no_grad(): 
        current_suffix_len = 0 
        for i in range(suffix_len): 
            if input_ids[0, orig_len + i] != bert_tokenizer.pad_token_id: 
                current_suffix_len = i + 1 
            else: 
                break

        last_token_id = input_ids[0, orig_len + current_suffix_len - 1] if current_suffix_len > 0 else None  
        is_last_connector = last_token_id in additional_connector_ids if last_token_id is not None else False  

        if additional_connector_ids and not is_last_connector and current_suffix_len < suffix_len:  
            connector_idx = additional_connector_ids.index(bert_tokenizer.encode("and", add_special_tokens=False)[0]) 
            input_ids[0, orig_len + current_suffix_len] = additional_connector_ids[connector_idx] 
            if seen_tokens is not None: 
                seen_tokens.add(additional_connector_ids[connector_idx]) 
            current_suffix_len += 1 

        remaining_space = suffix_len - current_suffix_len  # 计算剩余空间
        if remaining_space >= 2:  
            intensifier_idx = torch.randint(0, len(intensifier_ids), (1,)).item() 
            input_ids[0, orig_len + current_suffix_len] = intensifier_ids[intensifier_idx] 
            if seen_tokens is not None:  
                seen_tokens.add(intensifier_ids[intensifier_idx])
            current_suffix_len += 1  # 更新后缀长度

            available_tokens = candidate_ids if dataset_name == "AG-News" else [tid for tid in candidate_ids if tid not in seen_tokens] 
            if available_tokens: 
                new_token = available_tokens[torch.randint(0, len(available_tokens), (1,)).item()] 
                input_ids[0, orig_len + current_suffix_len] = new_token 
                if seen_tokens is not None:  #
                    seen_tokens.add(new_token) 

    return input_ids 

"""
功能概述：
optimize_adversarial_suffix 是 STAIN 系统的主函数，用于生成对抗性后缀并附加到原始提示后，误导大语言模型的预测。
它支持多种数据集（SST-2、StrategyQA、AG-News）
- 生成后缀 S = C ⊕ W（C 为连接短语，W 为对抗性词序列），通过迭代优化改变预测。
方法通过语义识别提取候选词，结合 top-k 采样和损失函数优化后缀，最终返回对抗性提示。
"""
def optimize_adversarial_suffix(
    model, 
    tokenizer,
    bert_model, 
    bert_tokenizer, 
    classifier_model, 
    classifier_tokenizer,
    orig_prompt: str,  
    target_label: int, 
    dataset_name: str,  
    suffix_len: int = 80,
    n_steps: int = 400, 
    batch_size: int = 10, 
    top_k: int = 50,  
    alpha: float = 1.0,  # 总损失中对抗性损失的权重
    beta: float = 0.5,   # 总损失中相似度损失的权重
    gamma: float = 0.1,  # 总损失中困惑度损失的权重
    use_model=None,
    ag_candidate_ids=None
) -> str:  # 返回优化后的对抗性提示文本
    device = classifier_model.device

    if tokenizer.pad_token is None: 
        if tokenizer.eos_token is not None: 
            tokenizer.pad_token = tokenizer.eos_token  
        else: 
            default_pad_token = '[PAD]' 
            tokenizer.add_special_tokens({'pad_token': default_pad_token}) 
            tokenizer.pad_token = default_pad_token  
        print(f"Set tokenizer.pad_token to {tokenizer.pad_token}") 

  
    if dataset_name == "AG-News":  
        adv_text = attack_for_agnews(  
            bert_model=bert_model, 
            classifier_model=classifier_model,  
            classifier_tokenizer=classifier_tokenizer,
            bert_tokenizer=bert_tokenizer, 
            input_text=orig_prompt,  
            target_label=target_label, 
            device=device, 
            max_step=30, 
            perplexity_threshold=5.0, 
            alpha=alpha,  # 传递对抗性损失权重
            beta=beta,  # 传递相似度损失权重
            gamma=gamma,  # 传递困惑度损失权重
            use_model=use_model, 
            model=model, 
            tokenizer=tokenizer,
            candidate_ids=ag_candidate_ids
        )
        return adv_text

    
    orig_ids = bert_tokenizer(orig_prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device) 
    orig_len = orig_ids.size(1) 

    category_ids, positive_ids, negative_ids = load_emotional_words(bert_tokenizer, dataset_name)
    candidate_ids = select_candidate_ids(dataset_name, target_label, category_ids, positive_ids, negative_ids) 

    
    connector_file = "connector_structures.json" 

    if not os.path.exists(connector_file):
        raise FileNotFoundError(f"Connector structures file not found at {connector_file}.") 
    with open(connector_file, "r") as f: 
        connector_structures = json.load(f) 
    print(f"Loaded {len(connector_structures)} connector structures.") 

    additional_connectors = ["and", "yet", "while", "or", "still", "somehow"] 
    additional_connector_ids = [bert_tokenizer.encode(word, add_special_tokens=False)[0] for word in additional_connectors] 

    intensifiers = ["utterly", "absolutely", "completely", "totally"] 
    intensifier_ids = [bert_tokenizer.encode(word, add_special_tokens=False)[0] for word in intensifiers] 

    
    prompt_embedding = use_model.encode(orig_prompt, convert_to_tensor=True)  
    connector_phrases = [phrase for phrase, _ in connector_structures] 
    connector_embeddings = use_model.encode(connector_phrases, convert_to_tensor=True)  
    similarities = torch.nn.functional.cosine_similarity(prompt_embedding, connector_embeddings, dim=-1) 

    # 选择相似度最高的连接短语
    structure_idx = torch.argmax(similarities).item()
    print(f"Selected connector phrase: '{connector_phrases[structure_idx]}' with similarity {similarities[structure_idx]:.4f}") 

    structure_name, structure_tokens = connector_structures[structure_idx]
    structure_token_ids = [bert_tokenizer.encode(word, add_special_tokens=False)[0] for word in structure_tokens]

    suffix_ids = structure_token_ids.copy()
    for _ in range(suffix_len - len(suffix_ids)): 
        suffix_ids.append(bert_tokenizer.pad_token_id) 
    suffix_ids = torch.tensor(suffix_ids, device=device).unsqueeze(0)  
    input_ids = torch.cat([orig_ids, suffix_ids], dim=1) 

    target_tensor = torch.tensor([target_label], device=device) 

    input_ids, seen_tokens = generate_suffix(
        bert_model=bert_model, 
        classifier_model=classifier_model, 
        classifier_tokenizer=classifier_tokenizer, 
        bert_tokenizer=bert_tokenizer, 
        input_ids=input_ids, 
        orig_len=orig_len, 
        suffix_len=suffix_len, 
        structure_token_ids=structure_token_ids, 
        candidate_ids=candidate_ids, 
        additional_connector_ids=additional_connector_ids, 
        target_tensor=target_tensor,  
        n_steps=n_steps, 
        top_k=top_k, 
        device=device, 
        dataset_name=dataset_name,  
        alpha=alpha,
        beta=beta, 
        gamma=gamma,  
        use_model=use_model,  
        model=model, 
        tokenizer=tokenizer 
    )

    input_ids = append_strong_suffix(
        input_ids, 
        orig_len, 
        suffix_len, 
        bert_tokenizer,  
        additional_connector_ids, 
        intensifier_ids, 
        candidate_ids,  
        seen_tokens, 
        dataset_name 
    )

    optimized_text = bert_tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return optimized_text


def generate_next_token(
    bert_model, 
    input_ids: torch.Tensor, 
    position: int,  
    top_k: int,
    seen_tokens: set, 
    bert_tokenizer, 
    candidate_ids: list, 
    temperature: float, 
    guide_weight: float = 0.0, 
    classifier_model=None,  
    classifier_tokenizer=None,
    target_tensor=None,  
    model=None, 
    tokenizer=None,
    total_loss_value: float = 0.0,

) -> int | None: 
    with torch.no_grad():  
        
        if model is not None and tokenizer is not None: 
            lm_inputs = input_ids[:, :position].clone() 
            outputs = model(lm_inputs) 
            logits = outputs.logits[0, -1, :] 
        else: 
            masked_ids = input_ids.clone() 
            masked_ids[0, position] = bert_tokenizer.mask_token_id 
            bert_inputs = { 
                "input_ids": masked_ids, 
                "attention_mask": torch.ones_like(masked_ids)
            }
            bert_inputs = {k: v.to(input_ids.device) for k, v in bert_inputs.items()} 
            outputs = bert_model(**bert_inputs) 
            logits = outputs.logits[0, position, :] 

        
        if model is not None and tokenizer is not None: 
            lm_probs = torch.softmax(logits, dim=-1)  
            lm_scores = lm_probs[candidate_ids] * 0.2 
        else: 
            lm_scores = torch.zeros(len(candidate_ids), device=input_ids.device) 
        logits_for_candidates = logits[candidate_ids]
        # 用 total_loss 调整得分
        total_loss_adjustment = -total_loss_value * 30.0  # 超参数 10.0，负值惩罚高 total_loss
        logits_for_candidates += lm_scores + total_loss_adjustment
        logits_for_candidates = logits_for_candidates / temperature
        _, top_k_indices = torch.topk(logits_for_candidates, min(top_k, len(candidate_ids)), dim=-1)
        idx = top_k_indices[torch.randint(0, len(top_k_indices), (1,)).item()]
        return candidate_ids[idx]

      
# def load_emotional_words_for_agnews(bert_tokenizer):
#     """
#     专门为 AG-News 数据集加载类别相关词汇
#     """
#     # 从数据集提取分类词汇，再找同义词扩充下，拿出去，不要写死在代码里
#     def get_words_by_category(category="world"):  # 定义内部函数，根据类别提取相关词汇
#         words = []  # 初始化空列表，用于存储筛选出的词汇
#         for pos in [wn.ADJ, wn.NOUN, wn.VERB]:  # 遍历三种词性：形容词、名词、动词
#             for synset in wn.all_synsets(pos):  # 遍历指定词性的所有同义词集
#                 word = synset.name().split('.')[0]  # 提取同义词集的词（去掉后缀，如“global.a.01”提取“global”）
#                 senti_synsets = list(swn.senti_synsets(word, pos[0]))  # 获取词的情感同义词集（根据词性）
#                 if senti_synsets:  # 如果存在情感同义词集
#                     s = senti_synsets[0]  # 取第一个情感同义词集
#                     if category == "world":  # 如果类别为“world”
#                         world_related = [  # 定义“world”类别相关词汇列表
#                             "global", "diplomatic", "international", "peaceful", "conflicting",
#                             "political", "governmental", "strategic", "regional", "geopolitical",
#                             "multinational", "sovereign", "cultural", "economic", "humanitarian",
#                             "democratic", "traditional", "social", "environmental", "historical",
#                             "internationalist", "ethnic", "national", "continental", "peacekeeping",
#                             "diplomatic", "cooperative", "unified", "diverse", "regulated",
#                             "nation", "government", "conflict", "peace", "treaty", "alliance",
#                             "war", "diplomacy", "culture", "economy", "society", "environment",
#                             "cooperate", "negotiate", "govern", "unite", "divide", "stabilize",
#                             "rebellion", "crisis", "revolution", "sanction", "migration",
#                             "border", "terrorism", "summit", "election", "protest",
#                             "invade", "occupy", "resist", "reform", "oppress", "liberate",
#                             "dictator", "regime", "uprising", "embargo", "refugee",
#                             "frontier", "insurgency", "conference", "vote", "demonstration",
#                             "attack", "defend", "challenge", "transform", "suppress", "emancipate"
#                         ]
#                         if word in world_related:  # 如果词在“world”相关词汇列表中
#                             words.append(word)  # 添加到词汇列表
#                     elif category == "sports":  # 如果类别为“sports”
#                         sports_related = [  # 定义“sports”类别相关词汇列表
#                             "exciting", "victorious", "thrilling", "dynamic", "competitive",
#                             "athletic", "energetic", "intense", "champion", "sportive",
#                             "spirited", "dramatic", "agile", "heroic", "triumphant",
#                             "powerful", "fast-paced", "strategic", "dedicated", "team-oriented",
#                             "athletic", "enduring", "fierce", "determined", "resilient",
#                             "coordinated", "disciplined", "motivated", "prestigious", "celebrated",
#                             "team", "player", "game", "match", "tournament", "championship",
#                             "athlete", "coach", "victory", "defeat", "score", "play",
#                             "compete", "win", "lose", "train", "celebrate", "perform",
#                             "stadium", "fans", "rival", "league", "season", "record",
#                             "medal", "olympics", "race", "goal", "strategy", "fitness",
#                             "kick", "shoot", "sprint", "jump", "defend", "attack",
#                             "arena", "crowd", "opponent", "division", "playoff", "achievement",
#                             "award", "worldcup", "marathon", "point", "tactic", "endurance",
#                             "pass", "strike", "dash", "leap", "block", "charge"
#                         ]
#                         if word in sports_related:  # 如果词在“sports”相关词汇列表中
#                             words.append(word)  # 添加到词汇列表
#                     elif category == "business":  # 如果类别为“business”
#                         business_related = [  # 定义“business”类别相关词汇列表
#                             "profitable", "risky", "successful", "innovative", "bankrupt",
#                             "corporate", "economic", "financial", "lucrative", "strategic",
#                             "commercial", "industrial", "entrepreneurial", "monetary", "prosperous",
#                             "expansive", "sustainable", "competitive", "global", "dynamic",
#                             "profitable", "thriving", "operational", "efficient", "growing",
#                             "leading", "established", "productive", "influential", "dominant",
#                             "company", "market", "profit", "investment", "finance", "industry",
#                             "business", "corporation", "startup", "economy", "trade", "growth",
#                             "invest", "expand", "profit", "manage", "innovate", "succeed",
#                             "bank", "stock", "share", "merger", "acquisition", "deal",
#                             "revenue", "budget", "contract", "sales", "marketing", "brand",
#                             "launch", "trade", "negotiate", "fund", "grow", "collapse",
#                             "enterprise", "portfolio", "equity", "partnership", "transaction",
#                             "income", "forecast", "agreement", "commerce", "advertising", "logo",
#                             "release", "exchange", "bargain", "finance", "scale", "fail"
#                         ]
#                         if word in business_related:  # 如果词在“business”相关词汇列表中
#                             words.append(word)  # 添加到词汇列表
#                     elif category == "scitech":  # 如果类别为“scitech”
#                         scitech_related = [  # 定义“scitech”类别相关词汇列表
#                             "innovative", "futuristic", "advanced", "technical", "smart",
#                             "scientific", "technological", "cutting-edge", "research", "digital",
#                             "automated", "experimental", "modern", "intelligent", "progressive",
#                             "revolutionary", "sophisticated", "data-driven", "virtual", "analytical",
#                             "automated", "pioneering", "computational", "electronic", "integrated",
#                             "optimized", "systematic", "futuristic", "adaptive", "breakthrough",
#                             "technology", "science", "research", "innovation", "device", "software",
#                             "data", "system", "network", "algorithm", "development", "experiment",
#                             "develop", "innovate", "research", "compute", "analyze", "engineer",
#                             "robot", "ai", "cloud", "security", "platform", "application",
#                             "internet", "code", "hardware", "machine", "learning", "discovery",
#                             "program", "design", "test", "deploy", "upgrade", "hack",
#                             "cyber", "quantum", "server", "encryption", "framework", "interface",
#                             "web", "script", "chip", "automation", "training", "invention",
#                             "build", "create", "simulate", "launch", "update", "breach"
#                         ]
#                         if word in scitech_related:  # 如果词在“scitech”相关词汇列表中
#                             words.append(word)  # 添加到词汇列表
#         return words  # 返回筛选出的词汇列表

#     world_words = get_words_by_category(category="world")  # 提取“world”类别相关词汇
#     sports_words = get_words_by_category(category="sports")  # 提取“sports”类别相关词汇
#     business_words = get_words_by_category(category="business")  # 提取“business”类别相关词汇
#     scitech_words = get_words_by_category(category="scitech")  # 提取“scitech”类别相关词汇
#     category_words = [world_words, sports_words, business_words, scitech_words]  # 构造类别词汇列表
#     category_ids = []  # 初始化类别词 ID 列表
#     for words in category_words:  # 遍历每个类别的词汇列表
#         ids = [bert_tokenizer.encode(word, add_special_tokens=False)[0] for word in words if bert_tokenizer.encode(word, add_special_tokens=False)]  # 将词汇编码为词元 ID
#         category_ids.append(list(set(ids)))  # 去重后添加到类别词 ID 列表
#     return category_ids  # 返回类别词 ID 列表（[world_ids, sports_ids, business_ids, scitech_ids]）