from experiments.constants import MAX_LENGTH, LABEL2ID
from datasets import Dataset, ClassLabel
from loguru import logger
import pandas as pd
import numpy as np

def encode_nli(example, tokenizer):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

def encode_contexts(example, tokenizer):
    return tokenizer(example["context"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

def loaddataset_splits():
    pass


def load_nli_dataset_from_df(df):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda examples: {"label": examples["gold_label"]}, batched=True)
    dataset = dataset.cast_column("label", ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction']))

    return dataset

def load_contexts_dataset_from_df(df):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda examples: {"label": examples["monotonicity"]}, batched=True)
    dataset = dataset.cast_column("label", ClassLabel(num_classes=2, names=['down', 'up']))

    return dataset

def encode_nli_dataset(dataset: Dataset, tokenizer):
    dataset = dataset.map(lambda example: encode_nli(example, tokenizer), batched=True)
    # dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "premise", "hypothesis",  "label"], device='cuda')
    return dataset 

def encode_contexts_dataset(dataset: Dataset, tokenizer):
    dataset = dataset.map(lambda example: encode_contexts(example, tokenizer), batched=True)
    # dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "premise", "hypothesis",  "label"], device='cuda')
    return dataset 

def rename_help_df(df):
    df = df.rename(columns={'sentence1': 'premise', 'sentence2': 'hypothesis'})
    df = df[['premise', 'hypothesis', 'gold_label']]
    return df

def load_help():
    help_full_path = './data/HELP/HELP_full.tsv'
    help_test_path = './data/HELP/HELP_test.tsv'

    test_df = pd.read_csv(help_test_path, sep='\t', header=0, index_col=0, on_bad_lines='skip', quotechar="'")
    logger.info(test_df.columns)

    # construct train by removing test from full
    train_df = pd.read_csv(help_full_path, sep='\t', header=0, index_col=0, on_bad_lines='skip', quotechar="'")
    logger.info(train_df.shape)

    train_df = pd.merge(train_df, test_df, on=['sentence1','sentence2', 'gold_label', 'gold_label_num'], how='left', indicator='Exist')
    train_df['Exist'] = np.where(train_df.Exist=="both", True, False)
    train_df = train_df.loc[~train_df.Exist]

    train_df = rename_help_df(train_df)
    test_df = rename_help_df(test_df)

    train_dataset = load_nli_dataset_from_df(train_df)
    test_dataset = load_nli_dataset_from_df(test_df)

    # train_dataset = train_dataset.map(lambda examples: {"labels": examples["gold_label"]}, batched=True)

    return train_dataset, test_dataset

def load_help_contexts():
    help_contexts_train_path = './data/HELP_Contexts/help_contexts_train.tsv' 
    help_contexts_test_path = './data/HELP_Contexts/help_contexts_test.tsv' 

    test_df = pd.read_csv(help_contexts_test_path, sep='\t', header=0, index_col=0, on_bad_lines='skip', quotechar="'")
    logger.info(test_df.columns)

    # construct train by removing test from full
    train_df = pd.read_csv(help_contexts_train_path, sep='\t', header=0, index_col=0, on_bad_lines='skip', quotechar="'")
    logger.info(train_df.shape)

    train_dataset = load_contexts_dataset_from_df(train_df)
    test_dataset = load_contexts_dataset_from_df(test_df)

    return train_dataset, test_dataset