import requests
import json

# 模型名称映射字典
MODEL_NAME_MAPPING = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3:8b",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral"
}

def get_prediction_call_model(model, text, dataset="AG-News"):
   
    # 假设 model 是字符串 (API 地址) 或对象
    # else 部分
    API_base = getattr(model, "API_base", "http://localhost:11434")
    version = getattr(model, "version", "llama3:8b")

    # 应用模型名称映射
    mapped_version = MODEL_NAME_MAPPING.get(version, version)
    if mapped_version != version:
        print(f"Mapping model name: {version} -> {mapped_version}")

    # 定制化 prompt 设计
    if dataset.lower() == "ag-news":
        prompt = f"Classify the following news text into one of these categories: [0: World, 1: Sports, 2: Business, 3: Sci/Tech]. Return only the number. Text: {text}"
    elif dataset.lower() == "sst2":
        prompt = f"Classify the sentiment of the movie review as positive (1) or negative (0). Return only a single digit (0 or 1), making a best-effort judgment based on available words if incomplete or ambiguous. Text: {text}"
    elif dataset.lower() == "strategyqa":
        prompt = f"Answer the following question with True or False based on reasoning. Return only True or False. Question: {text}"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

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
        
        # 解析响应
        result = data.get("response", data.get("text", ""))
        print(f"API raw Parsed Result: {result}") 
        if result:
            # 提取分类标签，尝试从整个字符串中找数字
            digits = "".join(filter(str.isdigit, result))
            if digits:
                label = int(digits[0])
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