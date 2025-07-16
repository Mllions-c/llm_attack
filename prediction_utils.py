import requests
import json
from openai import OpenAI
import os
from transformers import AutoTokenizer,AutoModelForCausalLM
# 模型名称映射字典
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL_NAME_MAPPING = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "gpt4",
    "mistralai/Mistral-7B-Instruct-v0.3": "gpt4",
    "EleutherAI/pythia-1.4b":"gpt4",
    "gpt2": "gpt4",
}

def get_prediction_call_model(model, text, dataset="AG-News"):
    """
    使用本地模型 API 预测文本的分类标签，针对不同数据集定制 prompt，并支持模型名称映射。
    
    参数:
        model: 模型接口，可以是 API 地址字符串或包含 API_base 和 version 的对象
        text: 待分类的对抗文本 (str)
        dataset: 数据集类型 ("AG-News", "sst2", "StrategyQA")
    
    返回:
        int: 预测的分类标签 (0, 1, 2, 3 for AG-News; 0 or 1 for sst2; 1 for True, 0 for False for StrategyQA)
    
    假设:
        - API 地址为 http://localhost:11434/api/generate
        - 返回 JSON 包含 'response' 或 'text' 字段
    """
    # 假设 model 是字符串 (API 地址) 或对象
    # else 部分
    

    # 应用模型名称映射
    mapped_version = MODEL_NAME_MAPPING.get(model, "llama3:8b")
    print(f"mapped_version: {mapped_version} dataset={dataset}")
    
    
    API_base = getattr(model, "API_base", "http://localhost:11434")
    # 定制化 prompt 设计
    if dataset.lower() == "ag-news":
        prompt = f"Classify the following news text into one of these categories: [0: World, 1: Sports, 2: Business, 3: Sci/Tech]. Return only the number. Text: {text}"
    elif dataset.lower() == "sst2":
        prompt = f"Classify the sentiment of the movie review as positive (1) or negative (0). Return only a single digit (0 or 1), making a best-effort judgment based on available words if incomplete or ambiguous. Text: {text}"
    elif dataset.lower() == "strategyqa":
        prompt = f"Answer the following question with True or False based on reasoning. Return only True or False. Question: {text}"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    if mapped_version=="gpt4":
        print("call gpt4")
        return get_prediction_call_online_model(prompt, dataset)
    # 设置请求参数
    payload = {
        "model": mapped_version,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7,
        "max_new_tokens": 10,
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        # 发送请求
        response = requests.post(f"{API_base}/api/generate", json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        #print(f"API Response: {data}")  # 调试：打印完整响应
        
        # 解析响应
        result = data.get("response", data.get("text", ""))
        print(f"API raw Parsed Result: {result}")  # 调试：打印解析后的 result
        if result:
            # 提取分类标签，尝试从整个字符串中找数字
            digits = "".join(filter(str.isdigit, result))
            if digits:
                label = int(digits[0])  # 取第一个数字
            elif dataset.lower() == "strategyqa":
                label = 1 if "true" in result.lower() else 0
            else:
                raise ValueError(f"No valid digit found in response: {result}")
        else:
            raise ValueError("No valid response from API")
        return label
            
    except requests.RequestException as e:
        print(f"API call failed: {e}")
        raise
    except ValueError as e:
        print(f"Failed to parse prediction: {e}")
        raise

def get_prediction_call_online_model(prompt, dataset="AG-News"):
        
    response = client.chat.completions.create(
        model="gpt-4",  # 或 "gpt-4o", "gpt-4.1-mini"，根据可用性选择
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,  # 限制输出长度
        temperature=0.3,  # 控制随机性
        top_p=0.5  # 核采样
    )

    # 解析响应
    decoded_text = response.choices[0].message.content.strip()
    print(f"Raw output: {decoded_text}")
    
    # 提取分类标签
    if dataset.lower() == "ag-news":
        digits = "".join(filter(str.isdigit, decoded_text))
        if digits:
            label = int(digits[0])  # 取第一个数字 (0-3)
        else:
            raise ValueError(f"No valid digit found in response: {decoded_text}")
    elif dataset.lower() == "sst2":
        digits = "".join(filter(str.isdigit, decoded_text))
        if digits:
            label = int(digits[0])  # 取第一个数字 (0 或 1)
        else:
            raise ValueError(f"No valid digit found in response: {decoded_text}")
    elif dataset.lower() == "strategyqa":
        if "true" in decoded_text.lower():
            label = 1
        elif "false" in decoded_text.lower():
            label = 0
        else:
            raise ValueError(f"No valid True/False found in response: {decoded_text}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return label