from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score
from data_utils import load_help_contexts, encode_contexts_dataset
from experiments.constants import MODEL_HANDLES, TOKENIZERS, CONTEXTS_LABEL2ID
from loguru import logger
import torch
import gc

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_score(y_true=labels, y_pred=predictions)



if __name__=='__main__':
    dataset_name = 'help_contexts'
    model_name = 'roberta-large-mnli'
    # device='cuda'

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZERS[model_name])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_HANDLES[model_name], num_labels=2, ignore_mismatched_sizes=True)

    train_dataset, eval_dataset = load_help_contexts()

    train_dataset = encode_contexts_dataset(train_dataset, tokenizer)
    train_dataset = train_dataset.align_labels_with_mapping(CONTEXTS_LABEL2ID, 'label')

    eval_dataset = encode_contexts_dataset(eval_dataset, tokenizer)
    eval_dataset = eval_dataset.align_labels_with_mapping(CONTEXTS_LABEL2ID, 'label')

    del tokenizer
    torch.cuda.empty_cache()

    training_args = TrainingArguments(output_dir="test_trainer", 
                                        evaluation_strategy="epoch",
                                        per_device_train_batch_size=1)

    # model.to(device)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)