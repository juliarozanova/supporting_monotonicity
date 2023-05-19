# Intro
This repo includes the data and finetuning scripts for the paper [``Supporting Context Monotonicity Abstraction in Neural NLI''](https://aclanthology.org/2021.naloma-1.6.pdf).

Please cite us as follows:

```
@inproceedings{rozanova-etal-2021-supporting,
    title = "Supporting Context Monotonicity Abstractions in Neural {NLI} Models",
    author = "Rozanova, Julia  and
      Ferreira, Deborah  and
      Thayaparan, Mokanarangan  and
      Valentino, Marco  and
      Freitas, Andr{\'e}",
    booktitle = "Proceedings of the 1st and 2nd Workshops on Natural Logic Meets Machine Learning (NALOMA)",
    month = jun,
    year = "2021",
    address = "Groningen, the Netherlands (online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naloma-1.6",
    pages = "41--50",
    abstract = "Natural language contexts display logical regularities with respect to substitutions of related concepts: these are captured in a functional order-theoretic property called monotonicity. For a certain class of NLI problems where the resulting entailment label depends only on the context monotonicity and the relation between the substituted concepts, we build on previous techniques that aim to improve the performance of NLI models for these problems, as consistent performance across both upward and downward monotone contexts still seems difficult to attain even for state of the art models. To this end, we reframe the problem of context monotonicity classification to make it compatible with transformer-based pre-trained NLI models and add this task to the training pipeline. Furthermore, we introduce a sound and complete simplified monotonicity logic formalism which describes our treatment of contexts as abstract units. Using the notions in our formalism, we adapt targeted challenge sets to investigate whether an intermediate context monotonicity classification task can aid NLI models{'} performance on examples exhibiting monotonicity reasoning.",
}
```

# Get Started

To reproduce the data preparation process, run all cells in the notebook data_prep/data_prep.ipynb

To rerun the training scripts, run the following
(adjusting the model_name variable as required in the respective files):

``` 
python -m experiments.finetune_help_contexts
```

```
python -m experiments.finetune_help
``` 

The hyperparameters BATCH_SIZE and MAX_LENGTH may be adjusted in the experiments/constants.py file.