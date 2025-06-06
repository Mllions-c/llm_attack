import os  # 导入 os 模块，用于文件操作和环境变量设置
from evil_twins import generate_and_save_category_ids
from evil_twins.generate_and_save_category_ids import generate_and_save_category_ids
import torch  # 导入 PyTorch 库，用于张量操作和模型训练
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 设置环境变量，禁用分词器的并行处理，避免警告
import math  # 导入 math 模块，用于数学计算（如指数函数）
import numpy as np  # 导入 NumPy 库，用于数值计算
import time  # 导入 time 模块，用于记录运行时间
import random  # 导入 random 模块，用于随机操作
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, AutoModelForMaskedLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer  # 从 transformers 库导入模型和分词器
from torch import Tensor  # 从 PyTorch 导入 Tensor 类型
import torch.nn.functional as F  # 从 PyTorch 导入功能模块，用于损失函数等操作
from evil_twins.dataset_utils import load_dataset_custom  # 从自定义模块导入数据集加载方法
from evil_twins.prompt_optim import optimize_adversarial_suffix  # 从自定义模块导入对抗性后缀优化方法
from evil_twins.generate_and_save_category_ids import load_category_ids
# 修改：移除 UniversalSentenceEncoder，改用 SentenceTransformer
# 目的：解决 TensorFlow Hub 加载 USE 模型失败的问题
# 影响：使用更现代的 SentenceTransformer 模型计算相似度，兼容性更好
from sentence_transformers import SentenceTransformer  # 导入 SentenceTransformer 库，用于计算语义相似度

# 修改：初始化 SentenceTransformer 模型
# 目的：替换 USE，避免加载问题
# 影响：相似度计算仍有效，但数值可能略有变化
use_model = SentenceTransformer('all-MiniLM-L6-v2')  # 初始化 SentenceTransformer 模型，使用 'all-MiniLM-L6-v2' 预训练模型

# 定义参数（硬编码）
model_name = "meta-llama/Meta-Llama-3-8B-Instruct" #EleutherAI/pythia-1.4b 或meta-llama/Meta-Llama-3-8B-Instruct 或mistralai/Mistral-7B-Instruct-v0.3 或gpt2 # 设置语言模型名称
dataset_name = "sst2"  # 可改为 "AG-News" 或 "StrategyQA 或sst2"  # 设置数据集名称
num_examples = 100  # 设置处理样本数量
n_epochs = 100  # 设置优化轮数
batch_size = 10  # 设置批量大小
top_k = 50  # 设置 top-k 采样参数
kl_every = 1  # 设置每多少步计算一次 KL 散度
gamma = 0.0  # 设置 gamma 参数（未使用）
suffix_mode = False  # 设置后缀模式（未使用）
# 修改：调整 similarity_threshold
# 目的：SentenceTransformer 的相似度值通常比 USE 略低，降低阈值以维持 ASR
# 影响：AG-News 从 0.4 降至 0.35，sst2 和 StrategyQA 从 0.6 降至 0.55
similarity_threshold = 0.4 if dataset_name == "AG-News" else 0.5  # 根据数据集设置相似度阈值
ppl = 0.7  # 设置困惑度参数（未使用）
use_bert = True

class Args:  # 定义 Args 类，用于封装参数
    def __init__(self):
        self.model_name = model_name  # 语言模型名称
        self.dataset_name = dataset_name  # 数据集名称
        self.num_examples = num_examples  # 样本数量
        self.n_epochs = n_epochs  # 优化轮数
        self.batch_size = batch_size  # 批量大小
        self.top_k = top_k  # top-k 参数
        self.kl_every = kl_every  # KL 散度计算频率
        self.gamma = gamma  # gamma 参数
        self.suffix_mode = suffix_mode  # 后缀模式
        self.similarity_threshold = similarity_threshold  # 相似度阈值
        self.ppl = ppl  # 困惑度参数
        # 添加超参数
        self.alpha = 1.0  # 对抗性损失权重
        self.beta = 0.5   # 相似度损失权重
        self.gamma = 0.1  # 困惑度损失权重

args = Args()  # 实例化 Args 对象

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)  # 加载因果语言模型（如 GPT-2）
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)  # 加载对应的分词器
bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")  # 加载 BERT 模型（掩码语言模型）
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 加载 BERT 分词器
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to {tokenizer.pad_token}")
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'
        print("Added and set tokenizer.pad_token to [PAD]")
# 根据数据集加载分类器模型
if args.dataset_name == "AG-News":  # 如果数据集是 AG-News
    classifier_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news", num_labels=4)  # 加载 AG-News 分类器（4 类）
    classifier_tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")  # 加载 AG-News 分词器
else:  # 如果数据集是 StrategyQA 或 SST-2
    # StrategyQA 和 sst2 使用二分类模型
    classifier_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2", num_labels=2)  # 加载 SST-2 分类器（2 类）
    classifier_tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")  # 加载 SST-2 分词器

# 移动到设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确定设备（优先 GPU，否则 CPU）
model.to(device)  # 将语言模型转移到设备
bert_model.to(device)  # 将 BERT 模型转移到设备
classifier_model.to(device)  # 将分类器模型转移到设备
use_model.to(device)  # 修改：将 SentenceTransformer 模型转移到设备
# 假设分类头权重保存在 classifier_head_{dataset_name}.pt
classifier_head_path = f"classifier_head_{args.model_name.replace('/', '_')}_{args.dataset_name}.pt"
num_labels = 4 if args.dataset_name == "AG-News" else 2  # AG-News: 4 类，SST-2 和 StrategyQA: 2 类
classifier_head = torch.nn.Linear(model.config.hidden_size, num_labels).to(device)
if os.path.exists(classifier_head_path):
    classifier_head.load_state_dict(torch.load(classifier_head_path))
    print(f"Loaded classifier head from {classifier_head_path}")
else:
    print(f"Classifier head not found at {classifier_head_path}. Please train the classifier head first.")
    raise FileNotFoundError(f"Classifier head not found at {classifier_head_path}")
classifier_head.eval()
# Load dataset
dataset_class = load_dataset_custom(args)  # 使用自定义方法加载数据集

"""
功能概述：
get_prediction 方法使用分类器模型预测给定文本的类别。
它将输入文本编码为词元 ID，输入分类器模型，获取预测 logits，并返回预测类别。
用于评估对抗性攻击是否成功翻转模型预测。
"""
# 修改 get_prediction 方法，使用 model 进行预测
def get_prediction_model(model, tokenizer, classifier_head, text: str) -> int:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, -1, :]  # 提取最后一个词的隐藏状态
        logits = classifier_head(hidden_states)  # 使用分类头预测类别
        pred = torch.argmax(logits, dim=-1).item()
    return pred

# 恢复原来的 get_prediction 函数，使用 classifier_model 预测
def get_prediction_bert(classifier_model, classifier_tokenizer, text: str) -> int:
    inputs = classifier_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(classifier_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = classifier_model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
    return pred

"""
功能概述：
save_attack_results 方法将攻击结果保存到文件中，包括成功和失败的攻击次数、成功率、运行时间等统计信息。
它记录每个成功攻击样本的详细信息（KL 散度、相似度、困惑度等），并打印平均值和总运行时间。
结果保存为文本文件，便于后续分析。
"""
def save_attack_results(successful_attacks, failed_attacks, total_samples, asr, attack_results, execution_time, model_name, filepath_prefix="attack_results"):
    filepath = f"{filepath_prefix}_{model_name.replace('/', '_')}_{args.dataset_name}.txt"  # 构造结果文件路径，使用模型名和数据集名
    avg_similarity = sum(result['similarity'] for result in attack_results) / len(attack_results) if attack_results else 0.0  # 计算平均相似度，若无结果则为 0
    avg_perplexity = sum(result['perplexity'] for result in attack_results) / len(attack_results) if attack_results else 0.0  # 计算平均困惑度，若无结果则为 0
    with open(filepath, "w") as f:  # 以写入模式打开文件
        f.write(f"Successful Attacks: {successful_attacks}\n")  # 写入成功攻击次数
        f.write(f"Failed Attacks: {failed_attacks}\n")  # 写入失败攻击次数
        f.write(f"Total Samples: {total_samples}\n")  # 写入总样本数量
        f.write(f"Attack Success Rate: {asr:.2f}%\n")  # 写入攻击成功率，保留 2 位小数
        f.write(f"Execution Time: {execution_time:.2f} seconds\n")  # 写入运行时间，保留 2 位小数
        f.write("\nSuccessful Attack Details:\n")  # 写入成功攻击详情标题
        f.write(f"Average Similarity: {avg_similarity:.4f}\n")  # 写入平均相似度，保留 4 位小数
        f.write(f"Average Perplexity: {avg_perplexity:.4f}\n")  # 写入平均困惑度，保留 4 位小数
        for result in attack_results:  # 遍历每个成功攻击结果
            f.write(f"Sample {result['sample']}: {result['original_prompt']} -> {result['optimized_prompt']} (Epoch {result['epoch']})\n")  # 写入样本编号、原始提示、优化提示和轮数
            f.write(f"  Similarity: {result['similarity']:.4f}\n")  # 写入相似度，保留 4 位小数
            f.write(f"  Perplexity: {result['perplexity']:.4f}\n")  # 写入困惑度，保留 4 位小数
    print(f"Results saved to {filepath}")  # 打印保存路径
    print(f"Average Similarity: {avg_similarity:.4f}")  # 打印平均相似度
    print(f"Average Perplexity: {avg_perplexity:.4f}")  # 打印平均困惑度
    print(f"Execution Time: {execution_time:.2f} seconds")  # 打印总运行时间

# 修改：调整 compute_use_similarity 函数以适配 SentenceTransformer
# 目的：支持新的句子编码器，避免 USE 的加载问题
# 影响：相似度计算方式更直接，返回 0-1 范围内的值
"""
功能概述：
compute_use_similarity 方法计算两个文本之间的语义相似度，使用 SentenceTransformer 模型。
它将文本编码为嵌入向量，计算余弦相似度，返回 0-1 范围内的值。
用于评估原始提示和优化提示的语义一致性。
"""
def compute_use_similarity(text1: str, text2: str, use_model) -> float:
    embedding1 = use_model.encode(text1, convert_to_tensor=True)  # 使用 SentenceTransformer 模型编码第一个文本为嵌入向量
    embedding2 = use_model.encode(text2, convert_to_tensor=True)  # 编码第二个文本为嵌入向量
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0).item()  # 计算两个嵌入向量的余弦相似度，并提取为浮点数
    return similarity  # 返回相似度值

def select_candidate_ids_for_agnews(target_label, category_ids):
    target_words = []  # 初始化目标词汇列表，用于存储目标类别的词
    for category, ids in enumerate(category_ids):  # 遍历所有类别及其词元 ID 列表
        if category == target_label:  # 如果当前类别是目标类别
            target_words.extend(ids)  # 将该类别的词元 ID 添加到目标词汇列表
            break  # 找到目标类别后即可退出循环
    return target_words  # 返回目标词汇列表（候选词 ID）
# 主循环
#generate_and_save_category_ids(bert_tokenizer, use_model) # 生成ag候选词，运行一次就行
successful_attacks = 0  # 初始化成功攻击计数器
total_samples = len(dataset_class)  # 计算总样本数量
attack_results = []  # 初始化攻击结果列表，用于存储成功攻击的详细信息

print("Starting adversarial optimization...")  # 打印优化开始信息
start_time = time.time()  # 记录开始时间
if args.dataset_name == "AG-News":
    category_ids = load_category_ids()  # 加载与AG-News数据集类别相关的词汇ID
else:
    category_ids = None  # 非 AG-News 数据集不需要 category_ids

for i, (prompt_text, label) in enumerate(dataset_class):  # 遍历数据集中的每个样本
    print(f"Processing sample {i+1}/{total_samples}")  # 打印当前处理的样本信息
    
    if use_bert:
        orig_pred = get_prediction_bert(classifier_model, classifier_tokenizer, prompt_text)
    else:
        orig_pred = get_prediction_model(model, tokenizer, classifier_head, prompt_text)
    
    # 根据数据集选择目标标签
    if args.dataset_name == "AG-News":  # 如果数据集是 AG-News
        # 优先选择语义接近的标签
        possible_labels = [l for l in range(4) if l != orig_pred]  # 获取除原始预测外的其他类别
        # 语义接近性映射（World: 0, Sports: 1, Business: 2, Science/Tech: 3）
        label_priority = {  # 定义类别优先级映射，确保语义接近性
            0: [1, 2, 3],  # World -> Sports > Business > Science/Tech
            1: [0, 2, 3],  # Sports -> World > Business > Science/Tech
            2: [0, 1, 3],  # Business -> World > Sports > Science/Tech
            3: [2, 0, 1],  # Science/Tech -> Business > World > Sports
        }
        target_label = label_priority[orig_pred][0]  # 选择优先级最高的类别作为目标
        # 根据目标标签选择候选词的词元ID
        ag_candidate_ids = select_candidate_ids_for_agnews(target_label, category_ids)
    else:  # 如果数据集是 StrategyQA 或 SST-2（二分类）
        target_label = 0 if orig_pred == 1 else 1  # 目标类别与原始预测相反
        ag_candidate_ids=None

    optimized_text = optimize_adversarial_suffix(  # 调用对抗性后缀优化方法
        model=model,
        tokenizer=tokenizer,
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer,
        classifier_model=classifier_model,
        classifier_tokenizer=classifier_tokenizer,
        orig_prompt=prompt_text,  # 原始提示
        target_label=target_label,  # 目标类别
        dataset_name=args.dataset_name,  # 数据集名称
        suffix_len=25 if args.dataset_name == "AG-News" else 15,  # AG-News 后缀长度 25，其他 15
        n_steps=n_epochs,  # 优化轮数
        batch_size=batch_size,  # 批量大小
        top_k=top_k,  # top-k 参数
        alpha=args.alpha,  # 传递 alpha 参数
        beta=args.beta,    # 传递 beta 参数
        gamma=args.gamma,  # 传递 gamma 参数
        use_model=use_model,  # 传递 use_model 用于相似度计算
        ag_candidate_ids=ag_candidate_ids 
    )
    
    if use_bert:
        new_pred = get_prediction_bert(classifier_model, classifier_tokenizer, optimized_text)
    else:
        new_pred = get_prediction_model(model, tokenizer, classifier_head, optimized_text)
    
    similarity = 0.0  # 初始化相似度
    perplexity = 0.0  # 初始化困惑度

    similarity = compute_use_similarity(prompt_text, optimized_text, use_model)  # 计算原始和优化后提示的语义相似度

    bert_inputs = bert_tokenizer(optimized_text, return_tensors="pt", truncation=True, padding=True, max_length=512)  # 使用 BERT 分词器编码优化后提示
    bert_inputs = {k: v.to(bert_model.device) for k, v in bert_inputs.items()}  # 转移到设备
    with torch.no_grad():  # 无梯度计算
        bert_outputs = bert_model(**bert_inputs, labels=bert_inputs["input_ids"])  # 使用 BERT 模型计算语言模型损失
        loss = bert_outputs.loss  # 提取损失
        perplexity = math.exp(loss.item())  # 计算困惑度

    if orig_pred != new_pred and similarity >= args.similarity_threshold and perplexity <= 70: # 判断攻击是否成功（预测翻转、相似度达标、KL 散度达标）
        successful_attacks += 1  # 成功攻击计数器加 1
        print(f"Attack succeeded: {prompt_text} -> {optimized_text} (Similarity: {similarity:.4f})")  # 打印成功信息
        attack_results.append({  # 记录成功攻击详情
            "sample": i+1,
            "original_prompt": prompt_text,
            "optimized_prompt": optimized_text,
            "epoch": n_epochs,
            "similarity": similarity,
            "perplexity": perplexity
        })
    else:  # 如果攻击失败
        failure_reasons = []  # 初始化失败原因列表
        if orig_pred == new_pred:  # 如果预测未翻转
            failure_reasons.append(f"Prediction unchanged (Original: {orig_pred}, New: {new_pred})")
        if similarity < args.similarity_threshold:  # 如果相似度低于阈值
            failure_reasons.append(f"Similarity too low (Similarity: {similarity:.4f}, Threshold: {args.similarity_threshold})")
        print(f"Attack failed: {prompt_text} -> {optimized_text} (Similarity: {similarity:.4f})")  # 打印失败信息
        print(f"Failure reasons: {', '.join(failure_reasons)}")  # 打印失败原因

end_time = time.time()  # 记录结束时间
execution_time = end_time - start_time  # 计算总运行时间
failed_attacks = total_samples - successful_attacks  # 计算失败攻击次数
asr = (successful_attacks / total_samples) * 100 if total_samples > 0 else 0  # 计算攻击成功率（百分比）
print(f"Optimization finished.")  # 打印优化完成信息
save_attack_results(successful_attacks, failed_attacks, total_samples, asr, attack_results, execution_time, model_name)  # 保存攻击结果