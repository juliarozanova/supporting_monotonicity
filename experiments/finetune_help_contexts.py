from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

from data_prep.data_utils import load_help_contexts, encode_contexts_dataset
from experiments.utils import compute_metrics
from experiments.constants import MODEL_HANDLES, TOKENIZERS, CONTEXTS_LABEL2ID
from loguru import logger
import torch
import argparse


def finetune_on_help_contexts(model_name, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZERS[model_name])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_HANDLES[model_name], num_labels=2, ignore_mismatched_sizes=True)

    model.config.n_labels = 2

    train_dataset, eval_dataset = load_help_contexts()

    train_dataset = encode_contexts_dataset(train_dataset, tokenizer)
    train_dataset = train_dataset.align_labels_with_mapping(CONTEXTS_LABEL2ID, 'label')

    eval_dataset = encode_contexts_dataset(eval_dataset, tokenizer)
    eval_dataset = eval_dataset.align_labels_with_mapping(CONTEXTS_LABEL2ID, 'label')

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

    # trainer.evaluate()

    trainer.train(resume_from_checkpoint=False)

    trainer.save_model(f'./models/{model_name}-help-contexts')

if __name__=='__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-bs", "--batch_size", type=int, help="Per-device batch size")

    args = argParser.parse_args()

    dataset_name = 'help_contexts'
    model_name = 'roberta-large-mnli'
    # device='cuda'
    finetune_on_help_contexts(model_name, args.batch_size)

    # output save?