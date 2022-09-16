from torch.utils.data import Dataset

MAXLEN = 64


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, model_type="unsup"):
        self.data = data
        self.tokenizer = tokenizer
        self.model_type = model_type

    def text2id(self, text):
        if self.model_type == "unsup":
            text_ids = self.tokenizer([text, text], max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
        elif self.model_type == "sup":
            text_ids = self.tokenizer([text[0], text[1], text[2]], max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')

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
        return self.text2id(self.data[index][0]), self.text2id(self.data[index][1]), int(self.data[index][2])
        