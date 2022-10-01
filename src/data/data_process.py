
import jsonlines


def load_data(name, path, model_type="unsup"):
    """根据名字加载不同的数据集"""
    def load_snli_data(path):
        with jsonlines.open(path, 'r') as f:
            if model_type == "unsup":
                return [line.get('origin') for line in f]
            elif model_type == "sup":
                return [(line['origin'], line['entailment'], line['contradiction']) for line in f]

    def load_lqcmc_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [line.strip().split('\t')[0] for line in f]    

    def load_sts_data(path):
        with open(path, 'r', encoding='utf8') as f:            
            return [(line.split("||")[1], line.split("||")[2], line.split("||")[3]) for line in f]
        
    assert name in ["snli", "lqcmc", "sts"]
    if name == 'snli':
        return load_snli_data(path)
    return load_lqcmc_data(path) if name == 'lqcmc' else load_sts_data(path) 
