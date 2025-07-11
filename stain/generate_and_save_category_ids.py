import json
import os
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import torch
import time

def load_emotional_words_for_agnews(bert_tokenizer, use_model, threshold=0.4, max_words=100):
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
    print(f"Generating category IDs for AG-News and saving to {filepath}...")
    start_time = time.time()
    category_ids = load_emotional_words_for_agnews(bert_tokenizer, use_model)
    print(f"Generated category IDs in {time.time() - start_time:.2f} seconds")
    category_ids_serializable = [list(ids) for ids in category_ids]
    save_start = time.time()
    with open(filepath, "w") as f:
        json.dump(category_ids_serializable, f, indent=4)
    print(f"Category IDs saved to {filepath} in {time.time() - save_start:.2f} seconds")
    print(f"Total time for generate_and_save_category_ids: {time.time() - start_time:.2f} seconds")

def load_category_ids(filepath="agnews_category_ids.json"):
    print(f"Loading category IDs from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Category IDs file not found at {filepath}. Please run generate_and_save_category_ids first.")
    with open(filepath, "r") as f:
        category_ids_serializable = json.load(f)
    category_ids = [ids for ids in category_ids_serializable]
    print(f"Category IDs loaded successfully.")
    return category_ids

def add_words_to_category_ids(bert_tokenizer, filepath="agnews_category_ids.json"):
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
        world_related,
        sports_related,
        business_related,
        scitech_related
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

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

nltk.download('wordnet')
nltk.download('sentiwordnet')

def load_emotional_words(bert_tokenizer, dataset_name, threshold=0.8, max_words=300):
    def get_words_by_category(category="positive"):
        words = []
        banned_words = {
            "ass", "bum", "bitch", "venom", "guts", "funky", "import", "unauthorized", "decor", "court", "color", "car",
            "male", "art", "disc", "ben", "del", "per", "alt", "gil", "com", "inc", "hal", "sub", "rec", "und", "una",
            "res", "sol", "dim", "non", "ina", "ago", "self", "au", "so", "ex", "ko", "gr", "well", "plus", "minus",
            "golden", "artistic", "banner", "barren", "modest", "underivative", "outer", "mild", "soft", "blue",
            "prolific", "incumbent", "pale", "rank", "due", "ill", "softhearted", "tall"
        }
        for synset in wn.all_synsets(wn.ADJ):
            word = synset.name().split('.')[0]
            if word in banned_words or len(word) <= 3:
                continue
            senti_synsets = list(swn.senti_synsets(word, 'a'))
            for s in senti_synsets:
                if category == "positive" and s.pos_score() > s.neg_score() and s.pos_score() >= threshold:
                    words.append(word)
                    break
                elif category == "negative" and s.neg_score() > s.pos_score() and s.neg_score() >= threshold:
                    words.append(word)
                    break
            if len(words) >= max_words:
                break
        return list(set(words))

    if dataset_name == "sst2":
        positive_words = [
            "great", "excellent", "positive", "wonderful", "cheerful", "terrific", "fantastic", "beautiful", "pretty",
            "better", "solid", "proper", "smooth", "ideal", "fortunate", "gorgeous", "pivotal", "brilliant", "fabulous",
            "charming", "outstanding", "superb", "amazing", "delightful", "splendid", "lovely", "marvelous", "enjoyable",
            "inspiring", "radiant", "admirable", "pleasing", "vibrant", "captivating", "heartwarming", "exhilarating",
            "refreshing", "uplifting", "stellar", "phenomenal", "glorious", "sensational", "remarkable", "thrilling",
            "divine", "enchanting", "spectacular", "impressive", "engaging", "enthralling", "mesmerizing", "stupendous",
            "exquisite", "blissful", "ecstatic", "joyful", "tremendous", "astounding", "magnificent", "awesome",
            "riveting", "dazzling", "incredible", "wonderous", "alluring", "enticing", "heartening", "euphoric",
            "breathtaking", "captivating", "exultant", "radiant", "jubilant", "exalted", "glorified", "invigorating",
            "enticing", "enlivening", "sparkling", "resplendent", "vivacious", "cheery", "buoyant", "ebullient",
            "effervescent", "jovial", "merry", "gleeful", "overjoyed", "rapturous", "thrilled", "elated", "exuberant",
            "felicitous", "gleaming", "luminous", "vital", "zestful", "dynamic", "spirited"
        ]
        negative_words = [
            "bad", "awful", "negative", "terrible", "sad", "poor", "dreadful", "dirty", "cold", "dead", "unable",
            "unused", "dependent", "chilling", "hideous", "creepy", "foolish", "painful", "pathetic", "miserable",
            "bleak", "grim", "horrible", "lousy", "disappointing", "depressing", "horrid", "mediocre", "dismal",
            "unpleasant", "tragic", "disturbing", "boring", "annoying", "frustrating", "tedious", "uninspired",
            "lackluster", "subpar", "abysmal", "atrocious", "ghastly", "wretched", "dire", "harrowing", "torturous",
            "gruesome", "appalling", "devastating", "dreary", "monotonous", "irritating", "unbearable", "excruciating",
            "insufferable", "deplorable", "repulsive", "disheartening", "lame", "despicable", "odious", "nauseating",
            "revolting", "abhorrent", "detestable", "loathsome", "vile", "heinous", "reprehensible", "disgusting",
            "repugnant", "offensive", "obnoxious", "infuriating", "exasperating", "aggravating", "maddening",
            "irksome", "vexing", "grating", "troublesome", "hateful", "shocking", "scandalous", "egregious",
            "intolerable", "unpalatable", "unsavory", "bitter", "galling", "oppressive", "stifling", "suffocating",
            "tormenting", "agonizing", "crushing", "heartbreaking"
        ]
    elif dataset_name == "StrategyQA":
        positive_words = [
            "correct", "accurate", "true", "valid", "right", "clear", "precise", "reliable", "trustworthy", "credible",
            "exact", "proper", "dependable", "authentic", "legitimate", "sound", "convincing", "confident", "certain",
            "definitive", "plausible", "rational", "logical", "coherent", "lucid", "cogent", "persuasive", "reasonable",
            "prudent", "sensible", "factual", "verifiable", "consistent", "trusty", "well-founded", "substantiated",
            "compelling", "assured", "veracious", "upright", "honest", "transparent", "unequivocal", "evident", "obvious",
            "indisputable", "irrefutable", "undeniable", "substantial", "credible", "accurate", "certified", "confirmed",
            "established", "proven", "validated", "trustable", "unassailable", "incontrovertible", "unquestionable",
            "dependable", "steadfast", "unerring", "faultless", "impeccable", "flawless", "secure", "stable", "solid",
            "assured", "decisive", "definitive", "explicit", "manifest", "patent", "pronounced", "prominent", "striking",
            "unmistakable", "veritable", "authentic", "bona_fide", "genuine", "real", "sincere", "truthful", "accurate",
            "correct", "just", "fair", "equitable", "impartial", "objective", "neutral", "balanced", "even-handed",
            "scrupulous", "conscientious", "principled"
        ]
        negative_words = [
            "wrong", "incorrect", "false", "invalid", "vague", "unreliable", "misleading", "confusing", "unclear",
            "faulty", "erroneous", "inaccurate", "dubious", "questionable", "deceptive", "flawed", "ambiguous",
            "uncertain", "imprecise", "inconsistent", "unconvincing", "baseless", "fallacious", "unfounded", "specious",
            "misguided", "absurd", "illogical", "irrational", "obscure", "unsubstantiated", "unpersuasive", "groundless",
            "unreasonable", "speculative", "deceitful", "misinformed", "errant", "untenable", "ridiculous", "ludicrous",
            "preposterous", "nonsensical", "misconceived", "unjustified", "unwarranted", "spurious", "fanciful",
            "implausible", "unbelievable", "fantastical", "far-fetched", "improbable", "doubtful", "suspect",
            "untrustworthy", "uncredible", "shaky", "flimsy", "weak", "feeble", "unstable", "unsound", "shoddy",
            "defective", "imperfect", "inadequate", "insufficient", "lacking", "deficient", "substandard", "poor",
            "inferior", "mediocre", "unsatisfactory", "unacceptable", "unfit", "inappropriate", "unsuitable",
            "incongruous", "incompatible", "contradictory", "paradoxical", "anomalous", "irregular", "aberrant",
            "deviant", "anomalous", "atypical", "unusual", "peculiar", "odd", "strange", "bizarre", "weird",
            "outlandish", "extravagant", "exaggerated", "overblown", "inflated"
        ]
    else:
        positive_words = []
        negative_words = []

    positive_words = list(set(positive_words + get_words_by_category(category="positive")))
    negative_words = list(set(negative_words + get_words_by_category(category="negative")))
    
    if len(positive_words) > max_words:
        positive_words = positive_words[:max_words]
    if len(negative_words) > max_words:
        negative_words = negative_words[:max_words]

    positive_ids = []
    negative_ids = []
    for word in positive_words:
        encoded = bert_tokenizer.encode(word, add_special_tokens=False)
        if encoded and len(encoded) == 1:
            positive_ids.append(encoded[0])
    for word in negative_words:
        encoded = bert_tokenizer.encode(word, add_special_tokens=False)
        if encoded and len(encoded) == 1:
            negative_ids.append(encoded[0])
    positive_ids = list(set(positive_ids))
    negative_ids = list(set(negative_ids))

    print(f"Positive IDs count: {len(positive_ids)}, Negative IDs count: {len(negative_ids)}")
    return [], positive_ids, negative_ids

def select_candidate_ids(dataset_name, target_label, positive_ids, negative_ids):
    if dataset_name == "sst2":
        return negative_ids if target_label == 0 else positive_ids
    else:
        return positive_ids if target_label == 1 else negative_ids

def convert_candidate_ids(candidate_ids, bert_tokenizer, tokenizer):
    candidate_words = [bert_tokenizer.decode([tid], skip_special_tokens=True).strip() for tid in candidate_ids]
    
    new_candidate_ids = []
    vocab = tokenizer.vocab
    skipped_reasons = {'empty': 0, 'multi_token': 0}
    
    for idx, (tid, word) in enumerate(zip(candidate_ids, candidate_words)):
        if not word:
            skipped_reasons['empty'] += 1
            continue
        if word in vocab:
            new_candidate_ids.append(vocab[word])
            continue
        encoded = tokenizer.encode(" " + word, add_special_tokens=False)
        if not encoded:
            skipped_reasons['empty'] += 1
            continue
        if len(encoded) > 1:
            skipped_reasons['multi_token'] += 1
            continue
        new_candidate_ids.append(encoded[0])
    
    print(f"Skipped reasons: {skipped_reasons},len:{len(new_candidate_ids)}")
    if not new_candidate_ids:
        print("Warning: No valid candidate IDs after conversion")
    
    return new_candidate_ids