import json
import os
from nltk.corpus import wordnet as wn
import torch
import time 

def load_emotional_words_for_agnews(bert_tokenizer, use_model, threshold=0.4, max_words=100):
    """
    专门为 AG-News 数据集加载类别相关词汇，通过语义相似度得分动态筛选

    参数：
        bert_tokenizer: BERT 分词器，用于将词汇编码为词元 ID
        use_model: SentenceTransformer 模型，用于计算语义相似度
        threshold: 相似度得分阈值，默认为 0.6
        max_words: 每个类别提取的最大词汇数量，默认为 100

    返回：
        category_ids: 类别词 ID 列表（[world_ids, sports_ids, business_ids, scitech_ids]）
    """
    def get_words_by_category(category="world"):
        print(f"Starting to process category: {category}") 
        start_time = time.time() 
        words = []
        max_similarity = float('-inf')
        max_similarity_word = None
        
        category_descriptions = {
            "world": "world news, global events, international politics",
            "sports": "sports, athletics, games, tournaments",
            "business": "business, finance, economics, markets",
            "scitech": "science, technology, innovation, research"
        }
        print(f"Encoding category description for {category}...") 
        category_start = time.time()
        category_embedding = use_model.encode(category_descriptions[category], convert_to_tensor=True)
        print(f"Category description encoding took {time.time() - category_start:.2f} seconds") 

        word_count = 0 
        for pos in [wn.ADJ, wn.NOUN, wn.VERB]: 
            print(f"Processing part of speech: {pos}")  
            pos_start = time.time()
            synsets = list(wn.all_synsets(pos)) 
            print(f"Loaded {len(synsets)} synsets for {pos} in {time.time() - pos_start:.2f} seconds")  

            for i, synset in enumerate(synsets): 
                if i % 1000 == 0: 
                    print(f"Processed {i} synsets for {pos}, found {len(words)} words so far") 
                word = synset.name().split('.')[0] 
                word_count += 1
                word_start = time.time()
                word_embedding = use_model.encode(word, convert_to_tensor=True)
                similarity = torch.nn.functional.cosine_similarity(word_embedding, category_embedding, dim=0).item()
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_similarity_word = word
                    print(f"New maximum similarity for {category}: '{max_similarity_word}' with similarity {max_similarity:.4f}")
                if similarity >= threshold:
                    words.append(word)
                    print(f"Added word '{word}' to {category} with similarity {similarity:.4f}") 
                if len(words) >= max_words: 
                    print(f"Reached max words ({max_words}) for {category}, stopping") 
                    break
            if len(words) >= max_words: 
                break
        print(f"Finished processing category {category}, found {len(words)} words in {time.time() - start_time:.2f} seconds")
        return words

    print("Starting to extract words for all categories...") 
    start_time = time.time()
    category_ids = []
    for category in ["world", "sports", "business", "scitech"]:
        words = get_words_by_category(category) 
        print(f"Encoding words for category {category}...")
        encode_start = time.time()
        ids = [bert_tokenizer.encode(word, add_special_tokens=False)[0] for word in words if bert_tokenizer.encode(word, add_special_tokens=False)]
        category_ids.append(list(set(ids)))
        print(f"Encoded {len(ids)} unique IDs for {category} in {time.time() - encode_start:.2f} seconds") 
    print(f"Finished extracting and encoding all categories in {time.time() - start_time:.2f} seconds") 
    return category_ids

def generate_and_save_category_ids(bert_tokenizer, use_model, filepath="agnews_category_ids.json"):
    """
    运行 load_emotional_words_for_agnews 并将结果保存到本地文件

    参数：
        bert_tokenizer: BERT 分词器
        use_model: SentenceTransformer 模型
        filepath: 保存路径，默认为 "agnews_category_ids.json"

    返回：
        None
    """
    print(f"Generating category IDs for AG-News and saving to {filepath}...")
    start_time = time.time()
    category_ids = load_emotional_words_for_agnews(bert_tokenizer, use_model) 
    print(f"Generated category IDs in {time.time() - start_time:.2f} seconds")
    
    category_ids_serializable = [list(ids) for ids in category_ids]
    
    save_start = time.time()
    with open(filepath, "w") as f:
        json.dump(category_ids_serializable, f, indent=4)
    print(f"Category IDs saved to {filepath} in {time.time() - save_start:.2f} seconds")  
    print(f"Total time for generate_and_save_category_ids: {time.time() - start_time:.2f} seconds")  # 日志：总耗时

def load_category_ids(filepath="agnews_category_ids.json"):
    """
    从本地文件加载 AG-News 类别词 ID

    参数：
        filepath: 加载路径，默认为 "agnews_category_ids.json"

    返回：
        category_ids: 类别词 ID 列表（[world_ids, sports_ids, business_ids, scitech_ids]）
    """
    print(f"Loading category IDs from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Category IDs file not found at {filepath}. Please run generate_and_save_category_ids first.")
    with open(filepath, "r") as f:
        category_ids_serializable = json.load(f)
    category_ids = [ids for ids in category_ids_serializable]
    print(f"Category IDs loaded successfully.")
    return category_ids


import json
import os
from nltk.corpus import wordnet as wn
import torch

def add_words_to_category_ids(bert_tokenizer, filepath="agnews_category_ids.json"):
    """
    将新的英文单词追加到 agnews_category_ids.json 中。

    参数：
        bert_tokenizer: BERT 分词器，用于将单词编码为词元 ID
        filepath: JSON 文件路径，默认为 "agnews_category_ids.json"
    """
    world_related = [
        "global", "diplomatic", "international", "peaceful", "conflicting",
        "political", "governmental", "strategic", "regional", "geopolitical",
        "multinational", "sovereign", "cultural", "economic", "humanitarian",
        "democratic", "traditional", "social", "environmental", "historical",
        "internationalist", "ethnic", "national", "continental", "peacekeeping",
        "diplomatic", "cooperative", "unified", "diverse", "regulated",
        "nation", "government", "conflict", "peace", "treaty", "alliance",
        "war", "diplomacy", "culture", "economy", "society", "environment",
        "cooperate", "negotiate", "govern", "unite", "divide", "stabilize",
        "rebellion", "crisis", "revolution", "sanction", "migration",
        "border", "terrorism", "summit", "election", "protest",
        "invade", "occupy", "resist", "reform", "oppress", "liberate",
        "dictator", "regime", "uprising", "embargo", "refugee",
        "frontier", "insurgency", "conference", "vote", "demonstration",
        "attack", "defend", "challenge", "transform", "suppress", "emancipate"
    ]
    sports_related = [
        "exciting", "victorious", "thrilling", "dynamic", "competitive",
        "athletic", "energetic", "intense", "champion", "sportive",
        "spirited", "dramatic", "agile", "heroic", "triumphant",
        "powerful", "fast-paced", "strategic", "dedicated", "team-oriented",
        "athletic", "enduring", "fierce", "determined", "resilient",
        "coordinated", "disciplined", "motivated", "prestigious", "celebrated",
        "team", "player", "game", "match", "tournament", "championship",
        "athlete", "coach", "victory", "defeat", "score", "play",
        "compete", "win", "lose", "train", "celebrate", "perform",
        "stadium", "fans", "rival", "league", "season", "record",
        "medal", "olympics", "race", "goal", "strategy", "fitness",
        "kick", "shoot", "sprint", "jump", "defend", "attack",
        "arena", "crowd", "opponent", "division", "playoff", "achievement",
        "award", "worldcup", "marathon", "point", "tactic", "endurance",
        "pass", "strike", "dash", "leap", "block", "charge"
    ]
    business_related = [
        "profitable", "risky", "successful", "innovative", "bankrupt",
        "corporate", "economic", "financial", "lucrative", "strategic",
        "commercial", "industrial", "entrepreneurial", "monetary", "prosperous",
        "expansive", "sustainable", "competitive", "global", "dynamic",
        "profitable", "thriving", "operational", "efficient", "growing",
        "leading", "established", "productive", "influential", "dominant",
        "company", "market", "profit", "investment", "finance", "industry",
        "business", "corporation", "startup", "economy", "trade", "growth",
        "invest", "expand", "profit", "manage", "innovate", "succeed",
        "bank", "stock", "share", "merger", "acquisition", "deal",
        "revenue", "budget", "contract", "sales", "marketing", "brand",
        "launch", "trade", "negotiate", "fund", "grow", "collapse",
        "enterprise", "portfolio", "equity", "partnership", "transaction",
        "income", "forecast", "agreement", "commerce", "advertising", "logo",
        "release", "exchange", "bargain", "finance", "scale", "fail"
    ]
    scitech_related = [
        "innovative", "futuristic", "advanced", "technical", "smart",
        "scientific", "technological", "cutting-edge", "research", "digital",
        "automated", "experimental", "modern", "intelligent", "progressive",
        "revolutionary", "sophisticated", "data-driven", "virtual", "analytical",
        "automated", "pioneering", "computational", "electronic", "integrated",
        "optimized", "systematic", "futuristic", "adaptive", "breakthrough",
        "technology", "science", "research", "innovation", "device", "software",
        "data", "system", "network", "algorithm", "development", "experiment",
        "develop", "innovate", "research", "compute", "analyze", "engineer",
        "robot", "ai", "cloud", "security", "platform", "application",
        "internet", "code", "hardware", "machine", "learning", "discovery",
        "program", "design", "test", "deploy", "upgrade", "hack",
        "cyber", "quantum", "server", "encryption", "framework", "interface",
        "web", "script", "chip", "automation", "training", "invention",
        "build", "create", "simulate", "launch", "update", "breach"
    ]

    new_words_by_category = [
        world_related,  # world
        sports_related,  # sports
        business_related,  # business
        scitech_related  # scitech
    ]

    print(f"Loading existing category IDs from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Category IDs file not found at {filepath}. Please run generate_and_save_category_ids first.")
    with open(filepath, "r") as f:
        category_ids = json.load(f)

    if len(category_ids) != 4:
        raise ValueError(f"Expected 4 categories in category_ids, but got {len(category_ids)}")

    for i, new_words in enumerate(new_words_by_category):
        new_ids = []
        for word in new_words:
            try:
                token_id = bert_tokenizer.encode(word, add_special_tokens=False)[0]
                new_ids.append(token_id)
            except Exception as e:
                print(f"Error encoding word '{word}' for category {i}: {str(e)}")
                continue
        category_ids[i].extend(new_ids)
        category_ids[i] = list(set(category_ids[i]))
        print(f"Updated category {i} with {len(new_ids)} new words, total unique IDs: {len(category_ids[i])}")

    print(f"Saving updated category IDs to {filepath}...")
    with open(filepath, "w") as f:
        json.dump(category_ids, f, indent=4)
    print(f"Category IDs updated successfully.")
