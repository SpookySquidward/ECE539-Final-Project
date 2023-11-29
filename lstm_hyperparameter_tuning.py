import torch
from torch import nn
import numpy as np
import model_runner
import embeddings
import dataset
import os
from models.lstm import review_LSTM
from save_load_model_parameters import file_exists
from wakepy import keep
from typing import Tuple, Any, OrderedDict


def construct_experiment(train_reviews: list[dataset.review],
                         val_reviews: list[dataset.review],
                         embedding_model: str = "glove-wiki-gigaword-50",
                         review_labels: list[str] = ["negative", "positive"],
                         oov_feature: bool = True,
                         title_body_feature: bool = True,
                         batch_size: int = 64,
                         lstm_layers: int = 1,
                         lstm_hidden_size: int = 100,
                         lstm_output_classifier: nn.Module = None,
                         lstm_output_classifier_name: str = "linear",
                         lr: int = 0.01,
                         ) -> Tuple[model_runner.runner,
                                    embeddings.batched_review_embedder_sampler,
                                    embeddings.batched_review_embedder_sampler]:
    
    # Model name (to save to disk)
    model_name_params = {
        "embedding-model": embedding_model,
        "oov": "Y" if oov_feature else "N",
        "title-body": "Y" if title_body_feature else "N",
        "batch-size": str(batch_size),
        "layers": str(lstm_layers),
        "hidden-size": str(lstm_hidden_size),
        "out-classifier": lstm_output_classifier_name,
        "lr": str(lr)
    }
    model_name = "tuning__" + "_".join("-".join((param_name, param_val)) for (param_name, param_val) in zip(model_name_params.keys(), model_name_params.values()))
    
    # Review embedder
    review_embedder = embeddings.review_embedder(review_labels, embedding_model, oov_feature, title_body_feature)

    # Samplers for the training and validation dataset
    train_sampler = embeddings.batched_review_embedder_sampler(train_reviews, review_embedder, batch_size)
    val_sampler = embeddings.batched_review_embedder_sampler(val_reviews, review_embedder, batch_size)
    
    # Use a linear LSTM output classifier if no classifier is specified
    if not lstm_output_classifier:
        lstm_output_classifier = nn.Sequential(nn.Linear(lstm_hidden_size * lstm_layers, len(review_labels)), nn.Softmax(dim=1))
    
    # Cumulative LSTM model
    lstm_input_size = review_embedder.feature_embedding_size
    model = review_LSTM(lstm_input_size, lstm_hidden_size, lstm_output_classifier, lstm_layers)
    
    # Optimization
    optim = torch.optim.Adam(model.parameters(), lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Model runner
    runner = model_runner.runner(model_name, model, optim, loss_fn)
    
    return runner, train_sampler, val_sampler


def permute_parameters_full(params: OrderedDict[str, list[Any]]):
    for param_key in params.values():
        for param_permutation in params[param_key]:
            pass # TODO


if __name__ == "__main__":
    print("All dependencies imported, preparing to tune hyperparameters...")
    
    # Reader to get train and val data from csv
    reader = dataset.csv_reader()
    
    # Open formatted_train.csv
    train_path = os.path.join(os.path.curdir, "dataset", "formatted_train.csv")
    reader.open_csv(train_path, skip_header=True)
    train_reviews = reader.read(-1)

    # And formatted_val.csv
    val_path = os.path.join(os.path.curdir, "dataset", "formatted_val.csv")
    reader.open_csv(val_path, skip_header=True)
    val_reviews = reader.read(-1)
    
    # Quick-and-dirty example for lr/batch_size
    with keep.running() as k:
        print("Successfully locked PC to prevent it sleeping during training!" if k.success else "Wasn't able to lock PC from sleeping during training!")
        
        for lr in [0.001, 0.003, 0.01, 0.03, 0.01]:
            for batch_size in [16, 32, 64, 128, 256]:
                print(f"\n\n\nTesting lr of {lr}, batch_size of {batch_size}...")
                runner, train_sampler, val_sampler = construct_experiment(train_reviews, val_reviews, lr=lr, batch_size=batch_size)
                
                if file_exists(runner._model_name):
                    print("Model already exists! This might not actually mean it was fully-trained... Skipping...")
                    continue
                
                runner.train(train_sampler, 10, val_sampler)
                
                print("Final train/val accuracy histories:", runner._train_acc_history, runner._val_acc_history, sep='\n')