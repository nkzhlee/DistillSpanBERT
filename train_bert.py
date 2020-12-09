# coding: utf-8

from __future__ import unicode_literals, print_function

import pandas as pd
import os
from transformers import BertTokenizer

from settings import bert_settings, ROOT_DATA_PATH, TRAIN_FILE
from bert_data import df_to_dataset
from bert_trainer import BertTrainer
from utils import get_logger, set_seed

if __name__ == '__main__':

    logger = get_logger()

    set_seed(3)

    train_df = pd.read_csv(TRAIN_FILE, encoding='utf-8', sep='\t')
    UNCASED = '/export/home/yug125/lzh/simple_version/bert-base-uncased/'  # your path for model and vocab
    VOCAB = 'bert-base-uncased-vocab.txt'
    bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED,VOCAB))
    train_dataset = df_to_dataset(train_df, bert_tokenizer, bert_settings['max_seq_length'])

    trainer = BertTrainer(bert_settings, logger)
    model = trainer.train(train_dataset, bert_tokenizer, ROOT_DATA_PATH)

