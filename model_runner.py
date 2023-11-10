import numpy as np
import torch
from torch import Tensor
from torch import nn
import save_load_model_parameters
from typing import Iterator, Tuple, Any
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os


class runner:
    
    # Training parameters to be saved and recalled
    param_key_model_state = "model_state"
    param_key_optimizer_state = "optimizer_state"
    param_key_epoch = "epoch"
    param_key_loss_history = "loss_history"
    param_key_train_acc_history = "train_acc_history"
    param_key_val_acc_history = "val_acc_history"
    
    # File name suffixes to be appended to model_name when saving the best and latest states
    file_name_suffix_latest = "_latest"
    file_name_suffix_previous = "_previous"
    file_name_suffix_best = "_best"
    
    
    def __init__(self, model_name: str, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss):
        self._model_name = model_name
        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        
        self._load_training_state(model_name + runner.file_name_suffix_latest)
    
    
    def state_dicts_are_compatible(dict1: dict[str, Any], dict2: dict[str, Any]) -> bool:
        # Check if all the keys are the same
        if dict1.keys() != dict2.keys():
            return False

        # Check that all the values are the same type
        for key in dict1.keys():
            if type(dict1[key]) != type(dict2[key]):
                return False
        
        # State dicts look compatible, return True
        return True
    
    
    def _load_training_state(self, file_name: str) -> bool:
        training_state = save_load_model_parameters.load_parameters(file_name)
        
        if training_state:
            # Save state found, load model parameters
            model_state = training_state[runner.param_key_model_state]
            optimizer_state = training_state[runner.param_key_optimizer_state]
            epoch = training_state[runner.param_key_epoch]
            loss_history = training_state[runner.param_key_loss_history]
            train_acc_history = training_state[runner.param_key_train_acc_history]
            val_acc_history = training_state[runner.param_key_val_acc_history]
            
            # Check that the model and optimizer state dicts appear compatible with the runner's
            # actual model and optimizer
            if (not runner.state_dicts_are_compatible(model_state, self._model.state_dict())):
                raise ValueError(f"Incompatible state dicts of local model ({self._model.state_dict()}) and loaded file ({model_state}); was the specified file_name correct?")
            if (not runner.state_dicts_are_compatible(optimizer_state, self._optimizer.state_dict())):
                raise ValueError(f"Incompatible state dicts of local optimizer ({self._optimizer.state_dict()}) and loaded file ({optimizer_state}); was the specified file_name correct?")
            
            # Apply the loaded state to the runner
            self._model.load_state_dict(model_state)
            self._optimizer.load_state_dict(optimizer_state)
            self._epoch = epoch
            self._loss_history = loss_history
            self._train_acc_history = train_acc_history
            self._val_acc_history = val_acc_history
            
            return True
        
        else:
            # Save state not found, initialize parameters to their default values (don't
            # re-initialize the model or optimizer, though)
            self._epoch = 0
            self._loss_history = []
            self._train_acc_history = []
            self._val_acc_history = []
            
            return False
    
    
    def _save_training_state(self, file_name: str, overwrite: bool = False) -> float:
        training_state = {
            runner.param_key_model_state: self._model.state_dict(),
            runner.param_key_optimizer_state: self._optimizer.state_dict(),
            runner.param_key_epoch: self._epoch,
            runner.param_key_loss_history: self._loss_history,
            runner.param_key_train_acc_history: self._train_acc_history,
            runner.param_key_val_acc_history: self._val_acc_history
        }
        
        save_load_model_parameters.save_parameters(training_state, file_name, overwrite)
        
    
    def _train_batch(self, x_batch: Any, y_batch: Tensor) -> Tuple[float, int, int]:
        # Reset gradients
        self._optimizer.zero_grad()
        
        # Forward pass
        yhat = self._model.forward(x_batch)
        
        # Loss
        loss = self._loss_fn(yhat, y_batch)
        loss.backward()
        
        # Optimization
        self._optimizer.step()
        
        # Training statistics
        # Number of samples which were trained upon
        num_samples = y_batch.shape[0]
        # Accurate predictions from those samples
        num_accurate_predictions = accuracy_score(runner.quantize_classifier_predictions(y_batch), runner.quantize_classifier_predictions(yhat))
        # Return training stats
        return (loss.item(), num_accurate_predictions, num_samples)
    
    
    def _train_epoch(self, train_batch_iterator: Iterator[Tuple[Any, Tensor]]) -> float:
        # Restart the iterator to begin a new epoch
        iter(train_batch_iterator)
        
        # Train each batch and track loss, training accuracy
        epoch_total_loss = 0.
        epoch_total_accurate_predictions = 0
        epoch_total_samples = 0
        
        # tqdm gives a nice status bar to show the epoch progress
        with tqdm(train_batch_iterator, f"Epoch {self._epoch + 1}", position=0, leave=True) as tepoch:
            for x_batch, y_batch in tepoch:
                # Batch train
                batch_loss, batch_accurate_predictions, batch_samples = self._train_batch(x_batch, y_batch)
                
                # Update statistics
                epoch_total_loss += batch_loss * batch_samples
                epoch_total_accurate_predictions += batch_accurate_predictions
                epoch_train_accuracy = epoch_total_accurate_predictions / epoch_total_samples if epoch_total_samples > 0 else 0
                epoch_total_samples += batch_samples
                
                # Update status bar to show current stats
                tepoch.set_postfix({"batch loss": f"{batch_loss:.3f}", "epoch train accuracy": f"{epoch_train_accuracy * 100:.1f}%"})
        
        # Update the epoch counter
        self._epoch += 1
        
        # Update the loss and test accuracy histories
        epoch_average_loss = epoch_total_loss / epoch_total_samples
        self._loss_history.append(epoch_average_loss)
        self._train_acc_history.append(epoch_train_accuracy)
                
    
    def train(self, train_batch_iterator: Iterator[Tuple[Any, Tensor]], val_batch_iterator: Iterator[Tuple[Any, Tensor]], num_epochs: int, autosave_interval_epochs: int = 1):
        starting_epoch = self._epoch
        for epoch in range(num_epochs):
            # Train the epoch
            self._train_epoch(train_batch_iterator)
            
            # Measure model accuracy against the validation dataset
            val_accuracy = self.classifier_accuracy_score(val_batch_iterator)
            self._val_acc_history.append(val_accuracy)
            
            # Find the most accurate (on the validation test set) model which has been trained so far
            most_accurate_model_epoch = np.argmax(self._val_acc_history)
            # If the current model is the most accurate one, save it to the disk, even if the
            # autosave interval hasn't been reached yet
            if most_accurate_model_epoch == self._epoch:
                self._save_training_state(self._model_name + runner.file_name_suffix_best, overwrite=True)
            
            # Save the latest runner state, if the autosave interval has been reached
            epochs_since_starting_training = self._epoch - starting_epoch
            if epochs_since_starting_training % autosave_interval_epochs == (autosave_interval_epochs - 1):
                self._save_training_state(self._model_name + runner.file_name_suffix_latest, overwrite=True)


    def predict_batch(self, batched_x: Any) -> Tensor:
        with torch.no_grad():
            y_hat = self._model.forward(batched_x)
            return y_hat
    
    
    def quantize_classifier_predictions(batched_predictions: Tensor) -> Tensor:
        return torch.argmax(batched_predictions, axis=1)
    
    
    def classifier_accuracy_score(self, batch_iterator: Iterator[Tuple[Any, Tensor]]) -> float:
        total_samples = 0
        correct_samples = 0
        
        for batched_x, batched_y in iter(batch_iterator):
            batched_y_hat = self.predict_batch(batched_x)
            correct_samples += accuracy_score(runner.quantize_classifier_predictions(batched_y), runner.quantize_classifier_predictions(batched_y_hat))
            total_samples += batched_y.shape[0]
        
        average_accuracy_score = correct_samples / total_samples
        return average_accuracy_score