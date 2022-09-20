import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
import torch.nn.functional as F
from .simcse_unsup import SimcseUnsupModel

from loguru import logger

class TrainUnsupSimcse:
    def __init__(self, pretrained_model_path, model_save_path) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_model_path = pretrained_model_path
        self.model_save_path = model_save_path
        self.best_loss = 1e8
        self.lr = 1e-5
        self.dropout = 0.3
        self.model = SimcseUnsupModel(pretrained_bert_path=self.pretrained_model_path, drop_out=self.dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    
    def simcse_unsup_loss(self, y_pred):
        y_true = torch.arange(y_pred.shape[0], device=self.device)
        y_true = (y_true - y_true % 2 * 2) + 1
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        sim = sim - torch.eye(y_pred.shape[0], device=self.device) * 1e12
        sim = sim / 0.05
        loss = F.cross_entropy(sim, y_true)
        return loss

    def train(self, train_dataloader, dev_dataloader):
        self.model.train()
        for batch_idx, source in enumerate(tqdm(train_dataloader), start=1):
            real_batch_num = source.get('input_ids').shape[0]
            input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(self.device)
            attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(self.device)
            token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(self.device)

            out = self.model(input_ids, attention_mask, token_type_ids)        
            loss = self.simcse_unsup_loss(out)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:     
                logger.info(f'loss: {loss.item():.4f}')
                corrcoef = self.eval(dev_dataloader)
                self.model.train()
                if self.best_loss > corrcoef:
                    self.best_loss = corrcoef
                    torch.save(self.model.state_dict(), self.model_save_path)
                    logger.info(f"higher corrcoef: {self.best_loss:.4f} in batch: {batch_idx}, save model")

    def eval(self, dataloader):
        self.model.eval()
        sim_tensor = torch.tensor([], device=self.device)
        label_array = np.array([])
        with torch.no_grad():
            for source, target, label in dataloader:
                source_input_ids = source.get('input_ids').squeeze(1).to(self.device)
                source_attention_mask = source.get('attention_mask').squeeze(1).to(self.device)
                source_token_type_ids = source.get('token_type_ids').squeeze(1).to(self.device)
                source_pred = self.model(source_input_ids, source_attention_mask, source_token_type_ids)

                target_input_ids = target.get('input_ids').squeeze(1).to(self.device)
                target_attention_mask = target.get('attention_mask').squeeze(1).to(self.device)
                target_token_type_ids = target.get('token_type_ids').squeeze(1).to(self.device)
                target_pred = self.model(target_input_ids, target_attention_mask, target_token_type_ids)
            
                sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
                sim_tensor = torch.cat((sim_tensor, sim), dim=0)            
                label_array = np.append(label_array, np.array(label))
        
        return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation