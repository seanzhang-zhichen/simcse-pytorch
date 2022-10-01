import os
import random
import argparse
from traceback import print_tb
from data.data_process import load_data, load_blog_data
from torch.utils.data import DataLoader, Dataset
from data.dataset import TrainDataset, TestDataset
from transformers import BertTokenizer

from model.train_unsup import TrainUnsupSimcse
from model.train_sup import TrainSupSimcse
from model.predict import get_sim
from loguru import logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型参数')
    parser.add_argument('--model_type', type=str, default='unsup', help='模型类型')
    args = parser.parse_args()
    batch_size = 64
    model_type = args.model_type
    logger.info(f"训练 {model_type} 模型...")

    bert_path = "../model/bert-base-chinese"
    model_save_path = f"../model/simcse/simcse_{model_type}.pt"
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    snli_train = '../data/cnsd-snli/train.txt'
    sts_train = '../data/STS-B/cnsd-sts-train.txt'
    sts_dev = '../data/STS-B/cnsd-sts-dev.txt'
    sts_test = '../data/STS-B/cnsd-sts-test.txt'


    if os.path.exists(model_save_path):
        logger.info(f"模型 {model_save_path} 已存在, 不训练...")
        text_a = "一个戴着草帽的人，站在外面用一堆椰子在外面工作"
        text_b = "一个人在烧草帽"
        text_c = "一个人在一堆椰子旁边"

        score_1 = get_sim(text_a, text_b, model_type)
        score_2 = get_sim(text_a, text_c, model_type)
        logger.info(f"text_a: {text_a}, text_b: {text_b}, simility: {score_1}")
        logger.info(f"text_a: {text_a}, text_b: {text_c}, simility: {score_2}")
        os._exit()

    

    if model_type == "unsup":
        train_data_snli = load_data('snli', snli_train, model_type)
        train_data_sts = load_data('sts', sts_train, model_type)
        train_data = train_data_snli + [_[0] for _ in train_data_sts]   # 两个数据集组合
        dev_data = load_data('sts', sts_dev, model_type)
        test_data = load_data('sts', sts_test, model_type)
        # train_data = load_blog_data()
    elif model_type == "sup":
        train_data = load_data('snli', snli_train, model_type)
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


    for epoch in range(1):
        logger.info(f'epoch: {epoch}')
        train_model.train(train_dataloader, dev_dataloader)
    logger.info(f'train is finished, best model is saved at {model_save_path}')

    dev_corrcoef = train_model.eval(dev_dataloader)
    test_corrcoef = train_model.eval(test_dataloader)
    logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
    logger.info(f'test_corrcoef: {test_corrcoef:.4f}')


