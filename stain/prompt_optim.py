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

def get_logits_with_embeddings(model, classifier_head, embeddings, max_length: int = 128):
    outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1][:, -1, :]
    logits = classifier_head(last_hidden_state)
    del outputs, last_hidden_state
    return logits

def get_prediction_model(model, tokenizer, classifier_head, text: str) -> int:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, -1, :]
        logits = classifier_head(hidden_states)
        pred = torch.argmax(logits, dim=-1).item()
    return pred

def attack_for_agnews(
    bert_model,
    classifier_model,
    classifier_tokenizer,
    bert_tokenizer,
    input_text: str,
    orig_pred: int,
    target_label: int,
    device,
    max_replacements: int = 30,
    similarity_threshold: float = 8.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
    use_model=None,
    model=None,
    tokenizer=None,
    classifier_head=None,
    candidate_ids=None,
    suffix_len: int = 20
):
    classifier_model.eval()
    bert_model.eval()
    model.eval()

    #编码
    inputs = bert_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    #连词处理
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
    
    # 选择相似度最高的连接短语
    structure_idx = torch.argmax(similarities).item()
    print(f"Selected connector phrase: '{connector_phrases[structure_idx]}' with similarity {similarities[structure_idx]:.4f}")
    structure_name, structure_tokens = connector_structures[structure_idx]
    structure_token_ids = [bert_tokenizer.encode(word, add_special_tokens=False)[0] for word in structure_tokens]

    seen_tokens = set(structure_token_ids)
    insert_ids = structure_token_ids.copy()

    # 对抗后缀初始化
    num_candidates = suffix_len
    available_tokens = [tid for tid in candidate_ids if tid not in seen_tokens]
    num_candidates = min(num_candidates, len(available_tokens))
    for _ in range(num_candidates):
        candidate_idx = torch.randint(0, len(available_tokens), (1,)).item()
        insert_ids.append(available_tokens[candidate_idx])
        seen_tokens.add(available_tokens[candidate_idx])
        available_tokens.pop(candidate_idx)

    #拼接成adv_input_ids
    orig_ids_list = input_ids[0].tolist()
    adv_input_ids = orig_ids_list[:-1] + insert_ids + [orig_ids_list[-1]]
    adv_input_ids = torch.tensor(adv_input_ids, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(adv_input_ids).to(device)

    
    replacements = 0
    # 主优化流程
    while replacements < max_replacements:
        current_text = bert_tokenizer.decode(adv_input_ids[0], skip_special_tokens=True)
        #困惑度
        ppl_loss = compute_perplexity_loss(bert_model, bert_tokenizer, current_text, device)
        perplexity = math.exp(ppl_loss.item())
        print(f"start Replacement {replacements+1} perplexity={perplexity}")
        
        #相似度
        sim_loss = 1.0 - compute_use_similarity(input_text, current_text, use_model)

        #adv_loss
        embeddings = model.get_input_embeddings()(adv_input_ids)
        embeddings.retain_grad()

        m_logits = get_logits_with_embeddings(model, classifier_head, embeddings=embeddings, max_length=128)
        target_tensor = torch.tensor([target_label], device=device)
        adv_loss = F.cross_entropy(m_logits, target_tensor)
        
        #total loss
        total_loss = alpha * adv_loss + beta * sim_loss + gamma * ppl_loss
        print(f"Iteration {replacements + 1}: adv_loss: {alpha * adv_loss:.4f}, sim_loss: {beta * sim_loss:.4f}, ppl_loss: {gamma * ppl_loss:.4f}, total_loss: {total_loss:.4f}")
        
        #初始化替换需要的参数
        gradients = torch.autograd.grad(outputs=total_loss, inputs=embeddings, retain_graph=False)[0]
        grad = gradients.detach()
        embedding_matrix = model.get_input_embeddings().weight.detach()
        best_loss = total_loss.item()
        best_pos = -1
        best_token = None
        suffix_start = len(orig_ids_list[:-1]) + len(structure_token_ids)
        suffix_end = len(orig_ids_list[:-1]) + len(structure_token_ids) + num_candidates
        max_end = adv_input_ids.size(1) - 1
        for pos in range(suffix_start, min(suffix_end, max_end)):
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
                # 替换
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_pos = pos
                    best_token = cand_id
        # 如果未找到更好的替换，停止循环
        if best_pos == -1:
            print("No better replacement found, stopping.")
            break

        adv_input_ids[0, best_pos] = best_token
        replacements += 1
        print(f"Replacement {replacements}: Position {best_pos}, New Token {bert_tokenizer.decode([best_token])}, Estimated Loss = {best_loss:.4f}")

        #反转检测
        current_text = bert_tokenizer.decode(adv_input_ids[0], skip_special_tokens=True)
        new_pred = get_prediction_model(model, tokenizer, classifier_head, current_text)
        similarity = compute_use_similarity(input_text, current_text, use_model)

        if new_pred != orig_pred and similarity > similarity_threshold:
            print("Prediction flipped successfully!")
            break

        if random.random() < 0.2:
            suffix_start = len(orig_ids_list[:-1]) + len(structure_token_ids)
            suffix_end = len(orig_ids_list[:-1]) + len(structure_token_ids) + num_candidates
            valid_positions = list(range(suffix_start, min(suffix_end, adv_input_ids.size(1) - 1)))

            if valid_positions:
                replace_pos = random.choice(valid_positions)
                current_token = adv_input_ids[0, replace_pos].item()
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
                        adv_input_ids[0, replace_pos] = new_token
                        seen_tokens.add(new_token)
                        print(f"Randomly replaced token at position {replace_pos} (POS: {target_pos}): '{current_word}' -> '{bert_tokenizer.decode([new_token])}'")
                    else:
                        print(f"No candidates with POS {target_pos} available for replacement at position {replace_pos}.")
                else:
                    print(f"Skipping replacement at position {replace_pos}: unsupported POS {current_pos} for '{current_word}'.")

    adv_text = bert_tokenizer.decode(adv_input_ids[0], skip_special_tokens=True)
    return adv_text

def compute_perplexity_loss(bert_model, bert_tokenizer, text, device):
    bert_model.eval()
    bert_inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    bert_inputs = {k: v.to(device) for k, v in bert_inputs.items()}
    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs, labels=bert_inputs["input_ids"])
        loss = bert_outputs.loss
    return loss

def compute_perplexity(bert_model, bert_tokenizer, text, device):
    bert_model.eval()
    bert_inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    bert_inputs = {k: v.to(device) for k, v in bert_inputs.items()}
    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs, labels=bert_inputs["input_ids"])
        loss = bert_outputs.loss
        perplexity = math.exp(loss.item())
    return perplexity

def compute_use_similarity(prompt, optimized_prompt, use_model):
    prompt_embedding = use_model.encode(prompt, convert_to_tensor=True)  # 使用 SentenceTransformer 模型编码原始提示为嵌入向量，转换为张量
    optimized_embedding = use_model.encode(optimized_prompt, convert_to_tensor=True)  # 编码优化后的提示为嵌入向量，转换为张量
    similarity = torch.nn.functional.cosine_similarity(prompt_embedding, optimized_embedding, dim=0).item()  # 计算两个嵌入向量的余弦相似度，并提取为浮点数
    return similarity  # 返回语义相似度值


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
optimize_adversarial_suffix 是 STAIN 系统的主函数，用于生成对抗性后缀并附加到原始提示后，误导大语言模型的预测。
它支持多种数据集（SST-2、StrategyQA、AG-News）
- 生成后缀 S = C ⊕ W（C 为连接短语，W 为对抗性词序列），通过迭代优化改变预测。
"""
def optimize_adversarial_suffix(
     model,
    tokenizer,
    bert_model,
    bert_tokenizer,
    classifier_model,
    classifier_tokenizer,
    orig_prompt: str,
    orig_pred:int,
    target_label: int,
    dataset_name: str,
    suffix_len: int = 20,
    similarity_threshold: float = 8.0,
    n_steps: int = 400,
    batch_size: int = 10,
    top_k: int = 50,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
    use_model=None,
    candidate_ids=None,
    classifier_head=None,  # 从主流程传递的分类头
) -> str:
    device = classifier_model.device

    # 设置tokenizer的pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            default_pad_token = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': default_pad_token})
            tokenizer.pad_token = default_pad_token
        print(f"Set tokenizer.pad_token to {tokenizer.pad_token}")

    adv_text = attack_for_agnews(
        bert_model=bert_model,
        classifier_model=classifier_model,
        classifier_tokenizer=classifier_tokenizer,
        bert_tokenizer=bert_tokenizer,
        input_text=orig_prompt,
        orig_pred=orig_pred,
        target_label=target_label,
        device=device,
        max_replacements=50 if dataset_name == "AG-News" else 50,
        similarity_threshold=similarity_threshold,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        use_model=use_model,
        model=model,
        tokenizer=tokenizer,
        classifier_head=classifier_head,  # 从主流程传递的分类头
        candidate_ids=candidate_ids,
        suffix_len=suffix_len
    )

    return adv_text