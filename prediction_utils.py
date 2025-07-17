import requests
import json
from openai import OpenAI
import os
from transformers import AutoTokenizer,AutoModelForCausalLM
# 模型名称映射字典
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL_NAME_MAPPING = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "meta-llama/llama-3.1-8b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistralai/mistral-7b-instruct-v0.3",
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
    # router
    if mapped_version=="gpt4":
        print("call gpt4")
        return get_prediction_call_online_model(prompt, dataset)
    elif mapped_version=="meta-llama/llama-3.1-8b-instruct" or mapped_version =="mistralai/mistral-7b-instruct-v0.3":
        print("call openrouter")
        return get_prediction_openrouter(prompt,dataset,mapped_version)
    # 设置请求参数
    payload = {
        "model": mapped_version,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7,
        "max_new_tokens": 5,
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
        return parse_label_from_response(result,dataset)
            
    except requests.RequestException as e:
        print(f"API call failed: {e}")
        raise
    except ValueError as e:
        print(f"Failed to parse prediction: {e}")
        raise

def get_prediction_call_online_model(prompt, dataset="AG-News"):
    max_retries = 3  # Maximum number of retries for invalid responses
    for attempt in range(max_retries):
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
        print(f"Raw output (attempt {attempt+1}): {decoded_text}")
        
        label=parse_label_from_response(decoded_text,dataset)
        if label !=None:
            return label
    
    # If all retries fail, return a default value or raise an error
    print(f"Skipping sample after {max_retries} failed attempts.")
    return None  # Or raise an error, adjust based on your needs


def get_prediction_openrouter(prompt, dataset, model):
   
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.7
    }
    try:
        max_retries = 3  # Maximum number of retries for invalid responses
        for attempt in range(max_retries):
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()  # 抛出 HTTP 错误
            result = response.json()["choices"][0]["message"]["content"]
            print(f"Raw output (attempt {attempt+1}): {result}")
            label=parse_label_from_response(result,dataset)
            if label !=None:
                return label
    except requests.RequestException as e:
        print(f"API call failed: {e}")
        raise
    except ValueError as e:
        print(f"Failed to parse prediction: {e}")
        raise
    
    return None

def parse_label_from_response(result, dataset):
    if not result:
        raise ValueError("No valid response from API")
    digits = "".join(filter(str.isdigit, result))
    if dataset.lower() in ["ag-news", "sst2"]:
        if digits:
            return int(digits[0])  # 取第一个数字 (0-3)
        raise ValueError(f"No valid digit found in response: {result}")
    elif dataset.lower() == "strategyqa":
        return 1 if "true" in result.lower() else 0
    else:
        raise ValueError(f"No valid digit found in response: {result}")