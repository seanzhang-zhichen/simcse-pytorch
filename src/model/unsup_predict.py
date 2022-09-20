
import torch
import numpy as np
import torch.nn.functional as F
from simcse_unsup import SimcseUnsupModel
from transformers import BertConfig, BertModel, BertTokenizer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def predict(tokenizer, model, text_a, text_b):

    token_a = tokenizer([text_a], max_length=64, truncation=True, padding='max_length', return_tensors='pt')
    token_b = tokenizer([text_b], max_length=64, truncation=True, padding='max_length', return_tensors='pt')
    model.eval()
    with torch.no_grad():    
        source_input_ids = token_a.get('input_ids').squeeze(1).to(DEVICE)
        source_attention_mask = token_a.get('attention_mask').squeeze(1).to(DEVICE)
        source_token_type_ids = token_a.get('token_type_ids').squeeze(1).to(DEVICE)
        source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
        # target        [batch, 1, seq_len] -> [batch, seq_len]
        target_input_ids = token_b.get('input_ids').squeeze(1).to(DEVICE)
        target_attention_mask = token_b.get('attention_mask').squeeze(1).to(DEVICE)
        target_token_type_ids = token_b.get('token_type_ids').squeeze(1).to(DEVICE)
        target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
        # concat
        sim = F.cosine_similarity(source_pred, target_pred, dim=-1).item()
        print(sim)


if __name__ == '__main__':
    text_a = "用len函数出现报错：索引超出矩阵范围"
    text_b = "c语言堆栈溢出"

    save_path = '../../model/simcse/simcse_unsup.pt'
    model_path = '../../model/bert-base-chinese'
    model = SimcseUnsupModel(pretrained_bert_path=model_path, drop_out=0.3).to(DEVICE)
    model.load_state_dict(torch.load(save_path))
    tokenizer = BertTokenizer.from_pretrained(model_path)
    predict(tokenizer, model, text_a, text_b)
