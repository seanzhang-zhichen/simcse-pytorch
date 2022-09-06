from torch.utils.data import DataLoader, Dataset

MAXLEN = 64


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def text2id(self, text):
        text_ids = self.tokenizer([text, text], max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
        return text_ids

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.text2id(self.data[index])


class TestDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def text2id(self, text):
        text_ids = self.tokenizer(text, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
        return text_ids

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.text2id(self.data[index][0]), self.text2id(self.data[index][1])
        