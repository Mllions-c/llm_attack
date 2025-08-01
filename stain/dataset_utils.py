from datasets import load_dataset

def load_dataset_custom(args):
    dataset_name = getattr(args, 'dataset_name', 'sst2')
    num_examples = getattr(args, 'num_examples', 10000) 
    
    # 数据集映射，方便扩展
    DATASET_LOCATION = {
        'sst2': 'stanfordnlp/sst2',
        'AG-News': 'ag_news',
        'StrategyQA': 'ChilleD/StrategyQA',
    }
    
    if dataset_name not in DATASET_LOCATION:
        raise ValueError(f"Dataset '{dataset_name}' not supported yet. Supported: {list(DATASET_LOCATION.keys())}")
    
    dataset_path = DATASET_LOCATION[dataset_name]
    print(f"Loading {dataset_name} dataset from {dataset_path}...")
    split = 'validation' if dataset_name == 'sst2' else 'test'
    dataset = load_dataset(dataset_path, split=split)
    
    # 简单取样，无复杂过滤，保留扩展性
    if dataset_name == 'sst2':
        dataset_class = [(d['sentence'], d['label']) for d in dataset][:num_examples]
    elif dataset_name == 'AG-News':
        dataset_class = [(d['text'], d['label']) for d in dataset][:num_examples]
    else:  # StrategyQA
        dataset_class = [(d['question'], 1 if d['answer'] else 0) for d in dataset][:num_examples]
    
    print(f"Loaded {len(dataset_class)} samples from {dataset_name}")
    return dataset_class