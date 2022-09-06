from data.data_process import load_data
from torch.utils.data import DataLoader, Dataset
from data.dataset import TrainDataset, TestDataset
from transformers import BertConfig, BertModel, BertTokenizer

from model.train import TrainUnsupSimcse

from loguru import logger


if __name__ == '__main__':
    batch_size = 64

    text = "今天天气真不错"
    bert_path = "../model/bert-base-chinese"
    model_save_path = "./model/simcse/simcse_unsup.pt"
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    snli_train = '../data/cnsd-snli/train.txt'
    sts_train = '../data/STS-B/cnsd-sts-train.txt'
    sts_dev = '../data/STS-B/cnsd-sts-dev.txt'
    sts_test = '../data/STS-B/cnsd-sts-test.txt'


    train_data_snli = load_data('snli', snli_train)
    train_data_sts = load_data('sts', sts_train)
    train_data = train_data_snli + [_[0] for _ in train_data_sts]   # 两个数据集组合
    dev_data = load_data('sts', sts_dev)
    test_data = load_data('sts', sts_test)


    train_dataloader = DataLoader(TrainDataset(train_data, tokenizer), batch_size=batch_size)
    dev_dataloader = DataLoader(TestDataset(dev_data, tokenizer), batch_size=batch_size)
    test_dataloader = DataLoader(TestDataset(test_data, tokenizer), batch_size=batch_size)

    train_model = TrainUnsupSimcse(bert_path, model_save_path)

    for epoch in range(10):
        logger.info(f'epoch: {epoch}')
        train_model.train(train_dataloader, dev_dataloader)
    logger.info(f'train is finished, best model is saved at {model_save_path}')

    dev_corrcoef = train_model.eval(dev_dataloader)
    test_corrcoef = train_model.eval(test_dataloader)
    logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
    logger.info(f'test_corrcoef: {test_corrcoef:.4f}')


