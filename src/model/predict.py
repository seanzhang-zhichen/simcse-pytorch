
import torch
import torch.nn.functional as F
from .simcse_unsup import SimcseUnsupModel
from .simcse_sup import SimcseSupModel
from transformers import BertTokenizer


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
        return sim


def get_sim(text_a, text_b, model_type):
    bert_path = '../model/bert-base-chinese'

    if model_type == "unsup":
        model_path = '../model/simcse/simcse_unsup.pt'
        model = SimcseUnsupModel(pretrained_bert_path=bert_path, drop_out=0.3).to(DEVICE)
    elif model_type == "sup":
        model_path = '../model/simcse/simcse_sup.pt'
        model = SimcseSupModel(pretrained_bert_path=bert_path, drop_out=0.3).to(DEVICE)

    model.load_state_dict(torch.load(model_path))
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    sim_score = predict(tokenizer, model, text_a, text_b)
    return sim_score