
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig



class SimcseSupModel(nn.Module):
    def __init__(self, pretrained_bert_path, drop_out) -> None:
        super(SimcseSupModel, self).__init__()

        self.pretrained_bert_path = pretrained_bert_path
        config = BertConfig.from_pretrained(self.pretrained_bert_path)
        config.attention_probs_dropout_prob = drop_out
        config.hidden_dropout_prob = drop_out
        self.bert = BertModel.from_pretrained(self.pretrained_bert_path, config=config)
    
    def forward(self, input_ids, attention_mask, token_type_ids, pooling="cls"):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if pooling == "cls":
            return out.last_hidden_state[:, 0]
        if pooling == "pooler":
            return out.pooler_output
        if pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)
            last = out.hidden_states[-1].transpose(1, 2)
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)


