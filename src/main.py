import random
from data.data_process import load_data
from torch.utils.data import DataLoader, Dataset
from data.dataset import TrainDataset, TestDataset
from transformers import BertConfig, BertModel, BertTokenizer

from model.train_unsup import TrainUnsupSimcse
from model.train_sup import TrainSupSimcse


from loguru import logger


if __name__ == '__main__':
    batch_size = 64

    text = "今天天气真不错"
    model_type = "sup"

    bert_path = "../model/bert-base-chinese"
    model_save_path = f"./model/simcse/simcse_{model_type}.pt"
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    snli_train = '../data/cnsd-snli/train.txt'
    sts_train = '../data/STS-B/cnsd-sts-train.txt'
    sts_dev = '../data/STS-B/cnsd-sts-dev.txt'
    sts_test = '../data/STS-B/cnsd-sts-test.txt'

    

    if model_type == "unsup":
        train_data_snli = load_data('snli', snli_train, model_type)
        train_data_sts = load_data('sts', sts_train, model_type)
        train_data = train_data_snli + [_[0] for _ in train_data_sts]   # 两个数据集组合
        dev_data = load_data('sts', sts_dev, model_type)
        test_data = load_data('sts', sts_test, model_type)
    elif model_type == "sup":
        train_data = load_data('snli', sts_train, model_type)
        random.shuffle(train_data)
        dev_data = load_data('sts', sts_dev, model_type)
        test_data = load_data('sts', sts_test, model_type)


    train_dataloader = DataLoader(TrainDataset(train_data, tokenizer, model_type=model_type), batch_size=batch_size)
    dev_dataloader = DataLoader(TestDataset(dev_data, tokenizer), batch_size=batch_size)
    test_dataloader = DataLoader(TestDataset(test_data, tokenizer), batch_size=batch_size)

    if model_type == "unsup":
        train_model = TrainUnsupSimcse(bert_path, model_save_path)
    elif model_type == "sup":
        train_model = TrainSupSimcse(bert_path, model_save_path)


    for epoch in range(10):
        logger.info(f'epoch: {epoch}')
        train_model.train(train_dataloader, dev_dataloader)
    logger.info(f'train is finished, best model is saved at {model_save_path}')

    dev_corrcoef = train_model.eval(dev_dataloader)
    test_corrcoef = train_model.eval(test_dataloader)
    logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
    logger.info(f'test_corrcoef: {test_corrcoef:.4f}')


