from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score
from data_utils import load_help, encode_nli_dataset
from experiments.constants import MODEL_HANDLES, TOKENIZERS, LABEL2ID_2CLASS
from loguru import logger
import torch
import argparse

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_score(y_true=labels, y_pred=predictions)

def finetune_on_help(model_name, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZERS[model_name])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_HANDLES[model_name], num_labels=2, ignore_mismatched_sizes=True)

    model.config.n_labels = 2

    train_dataset, eval_dataset = load_help()

    train_dataset = encode_nli_dataset(train_dataset, tokenizer)
    train_dataset = train_dataset.align_labels_with_mapping(LABEL2ID_2CLASS, 'label')

    eval_dataset = encode_nli_dataset(eval_dataset, tokenizer)
    eval_dataset = eval_dataset.align_labels_with_mapping(LABEL2ID_2CLASS, 'label')

    del tokenizer
    torch.cuda.empty_cache()

    training_args = TrainingArguments(output_dir="test_trainer", 
                                        evaluation_strategy="epoch",
                                        per_device_train_batch_size=batch_size)

    # model.to(device)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)

    trainer.save_model(f'./models/{model_name}-help')

if __name__=='__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-bs", "--batch_size", type=int, help="Per-device batch size")

    args = argParser.parse_args()

    dataset_name = 'help'
    model_name = 'roberta-large-mnli-help-contexts'

    # device='cuda'
    finetune_on_help(model_name, args.batch_size)