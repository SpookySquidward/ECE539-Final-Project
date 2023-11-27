import numpy as np
import torch
from torch import Tensor
from torch import nn
import save_load_model_parameters
from typing import Iterable, Tuple, Any
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


class runner:
    """Model runner class which can train a model, evaluate its performance, and save its state and performance history
    automatically
    """    
    
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
    
    
    def __init__(self, model_name: str, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.modules.loss._Loss, device: str = None) -> None:
        """Initializes the model runner.

        Args:
            model_name (str): The name of the model to be trained. Note that this name must be unique to a specific
            model structure, as it is used to determine the file names of save-states written to and read from
            `./model_parameters` (see `runner._load_training_state`).
            model (nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer used to train the given `model`.
            loss_fn (nn.modules.loss._Loss): The loss function used to evaluate the `model`'s performance.
            device (str, optional): The device to train the `model` on, e.g. `cuda` or `cpu`. If None, `cuda` is
            automatically selected for training, if it is available; otherwise `cpu` is used. Defaults to None.

        Raises:
            ValueError: If, when the `runner` is initialized, a save-state for `model` or `optimizer` is found in the
            temporary `./model_parameters` directory which has a matching `model_name` but a mismatching model or
            optimizer structure. See `runner._load_training_state`.
        """        
        
        # Model name is used to save checkpoints
        self._model_name = model_name
        
        # Training parameters
        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        
        # Assign a training device automatically if not specified
        if not device:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._device = device
        # Move the model to the target device
        self._model.to(self._device)
        
        # Load any existing checkpoints
        self._load_training_state(model_name + runner.file_name_suffix_latest)
    
    
    def state_dicts_are_compatible(dict1: dict[str, Any], dict2: dict[str, Any]) -> bool:
        """Checks to see whether two state dicts appear to be compatible with one another. State dicts are judged as
        compatible if their keys are identical and their corresponding values are of the same type.

        Args:
            dict1 (dict[str, Any]): The first state dict to compare.
            dict2 (dict[str, Any]): The second state dict to compare.

        Returns:
            bool: True if `dict1` and `dict2` are found to be compatible as described above, otherwise False.
        """        
        
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
        """Updates the `runner`'s model state, optimizer state, current epoch, and histories for train accuracy,
        validation accuracy, and train loss from the specified `file_name`. If no save file was found with the specified
        `file_name`, the above `runner`'s model an optimizer parameters remain unchanged, the epoch is initialized to
        zero, and the accuracy and loss histories are initialized to none.

        Args:
            file_name (str): The name of the file to load the `runner`'s states from. NOTE: this should be a file name,
            not a path from the root directory; parameters are automatically saved to the `./model_parameters` folder! A
            file extension of `.pt` is added automatically if not included in `file_name`. See
            `save_load_model_parameters.load_parameters`.

        Raises:
            ValueError: If a runner state is found with the specified `file_name`, but the model or optimizer state(s)
            contained within are not compatible with the `runner`'s currently-loaded model or optimizer.

        Returns:
            bool: True if the `runner`'s state was successfully updated from the specified `file_name`, or False if no
            save state at the specified `file_name` was found and the `runner`'s state was (re)initialized.
        """        
        
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
    
    
    def _save_training_state(self, file_name: str, overwrite: bool = False) -> bool:
        """Saves the `runner`'s model state, optimizer state, current epoch, and histories for train accuracy,
        validation accuracy, and train loss to the specified `file_name`. If a file with the specified `file_name`
        already exists and `overwrite` is False, no parameters are saved and False is returned.

        Args:
            file_name (str): The name of the file to save the `runner`'s states to. NOTE: this should be a file name,
            not a path from the root directory; parameters are automatically saved to the `./model_parameters` folder! A
            file extension of `.pt` is added automatically if not included in `file_name`. See
            `save_load_model_parameters.save_parameters`.
            overwrite (bool, optional): If True, overwrites any existing `runner` state with a matching `file_name`, if
            it exists (see `save_load_model_parameters.save_parameters`). Defaults to False.

        Returns:
            bool: Whether or not the `runner`'s training state was successfully saved to the specified `file_name`.
            False only if an existing model parameter with a matching `file_name` exists in the `./model_parameters`
            folder and `overwrite` is set to False (see `save_load_model_parameters.save_parameters`).
        """        
        
        training_state = {
            runner.param_key_model_state: self._model.state_dict(),
            runner.param_key_optimizer_state: self._optimizer.state_dict(),
            runner.param_key_epoch: self._epoch,
            runner.param_key_loss_history: self._loss_history,
            runner.param_key_train_acc_history: self._train_acc_history,
            runner.param_key_val_acc_history: self._val_acc_history
        }
        
        return save_load_model_parameters.save_parameters(training_state, file_name, overwrite)
        
    
    def _train_batch(self, x_batch: Any, y_batch: Tensor) -> Tuple[float, int, int]:
        """Trains the `runner`'s model on one batch of data.

        Args:
            x_batch (Any): The data to trian the `runner`'s model on. The type must be compatible with the model with
            which the `runner` was initialized.
            y_batch (Tensor): The corresponding labels for the specified `x_batch`. Labels must be of the shape
            `(N,*)`, where `N` is the number of samples in the current batch and `*` is any dimension.

        Returns:
            Tuple[float, int, int]: Batch training statistics of the form `(batch_loss, batch_num_accurate_predictions,
            batch_num_samples)`
        """        
        
        # Move batch data to the target device
        x_batch = x_batch.to(self._device)
        y_batch = y_batch.to(self._device)
        
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
        num_accurate_predictions = accuracy_score(runner.quantize_classifier_predictions(y_batch), runner.quantize_classifier_predictions(yhat), normalize=False)
        # Return training stats
        return (loss.item(), num_accurate_predictions, num_samples)
    
    
    def _train_epoch(self, train_batch_iterable: Iterable[Tuple[Any, Tensor]]) -> None:
        """Trains the `runner`'s model on one epoch of data.

        Args:
            train_batch_iterable (Iterable[Tuple[Any, Tensor]]): an iterable batch data loader which yields tuples of
            batch training data of the form `(x_batch, y_batch)`. The `x_batch` and `y_batch` data for each batch must
            conform to the structure specified in `runner._train_batch`: `x_batch` must be compatible with the model
            that the `runner` was initialized with, and `y_batch` must be of the shape `(N,*)`, where `N` is the number
            of samples in the batch and `*` is any dimension.
        """        
        
        
        # Restart the iterator to begin a new epoch
        train_batch_iterator = iter(train_batch_iterable)
        
        # Train each batch and track loss, training accuracy
        epoch_total_loss = 0.
        epoch_total_accurate_predictions = 0
        epoch_total_samples = 0
        
        # tqdm gives a nice status bar to show the epoch progress
        with tqdm(train_batch_iterator, f"Training Epoch {self._epoch + 1}", position=0, leave=True, unit="batches") as tepoch:
            for x_batch, y_batch in tepoch:
                # Batch train
                batch_loss, batch_accurate_predictions, batch_samples = self._train_batch(x_batch, y_batch)
                
                # Update statistics
                epoch_total_loss += batch_loss * batch_samples
                epoch_total_accurate_predictions += batch_accurate_predictions
                epoch_train_accuracy = epoch_total_accurate_predictions / epoch_total_samples if epoch_total_samples > 0 else 0
                epoch_total_samples += batch_samples
                
                # Update status bar to show current stats
                tepoch.set_postfix({"batch loss": f"{batch_loss:.3f}", "epoch train accuracy": f"{epoch_train_accuracy * 100:.2f}%"}, refresh=False)
        
        # Update the epoch counter
        self._epoch += 1
        
        # Update the loss and test accuracy histories
        epoch_average_loss = epoch_total_loss / epoch_total_samples
        self._loss_history.append(epoch_average_loss)
        self._train_acc_history.append(epoch_train_accuracy)
                
    
    def train(self, train_batch_iterable: Iterable[Tuple[Any, Tensor]],  num_epochs: int, val_batch_iterable: Iterable[Tuple[Any, Tensor]] = None, autosave_interval_epochs: int = 1):
        
        """Trains the `runner`'s model using a training and validaiton dataset, and collects statistics about the
        model's loss, train accuracy, and validation accuracy histories.

        Args:
            train_batch_iterable (Iterable[Tuple[Any, Tensor]]): an iterable batch data loader which yields tuples of
            batch training data of the form `(x_batch, y_batch)`. The `x_batch` and `y_batch` data for each batch must
            conform to the structure specified in `runner._train_batch`: `x_batch` must be compatible with the model
            that the `runner` was initialized with, and `y_batch` must be of the shape `(N,*)`, where `N` is the number
            of samples in the batch and `*` is any dimension.
            num_epochs (int): The number of epochs to train before halting.
            val_batch_iterable (Iterable[Tuple[Any, Tensor]], optional): an iterable batch data loader which yields
            tuples of batch validation data of the form `(x_batch, y_batch)`. The `x_batch` and `y_batch` data for each
            batch must conform to the structure specified in `runner.classifier_accuracy_score`: ``x_batch` must be
            compatible with the model that the `runner` was initialized with, and `y_batch` must be of the shape
            `(N,M)`, where `N` is the number of samples in the batch and `M` is the number of unique classes that the
            classifier can predict. If specified, the model's performance will be evaluated using this loader at the end
            of every epoch in order to always save the most accurate model state; otherwise, the model state will only
            be saved every `autosave_interval_epochs` epochs. Defaults to None.
            autosave_interval_epochs (int, optional): The number of epochs to train before saving the `runner`'s state
            with `runner._save_training_state`. Regardless of this value, if a model is found to have the highest
            validation accuracy so far after training an epoch, the `runner` state will be saved. Defaults to 1.
        """
        
        # Track epochs of training
        starting_epoch = self._epoch
        
        for epoch in range(num_epochs):
            # Train the epoch
            self._train_epoch(train_batch_iterable)
            
            if val_batch_iterable is not None:
                # Measure model accuracy against the validation dataset
                val_accuracy = self.classifier_accuracy_score(val_batch_iterable)
                self._val_acc_history.append(val_accuracy)
                
                # Find the most accurate (on the validation test set) model which has been trained so far
                val_acc_history_no_None = list((acc if acc is not None else -1) for acc in self._val_acc_history)
                most_accurate_model_epoch = np.argmax(val_acc_history_no_None) + 1
                
                # If the current model is the most accurate one, save it to the disk, even if the
                # autosave interval hasn't been reached yet
                if most_accurate_model_epoch == self._epoch:
                    print(f"This epoch was the most accurate so far: validation accuracy = {np.max(val_acc_history_no_None) * 100:.2f}%. Saving model state...")
                    self._save_training_state(self._model_name + runner.file_name_suffix_best, overwrite=True)
            
            else:
                # No validation dataset specified, skip validation
                self._val_acc_history.append(None)
            
            # Save the latest runner state, if the autosave interval has been reached
            epochs_since_starting_training = self._epoch - starting_epoch
            if epochs_since_starting_training % autosave_interval_epochs == 0 or epochs_since_starting_training == num_epochs:
                print("Reached epoch save interval, saving model state...")
                self._save_training_state(self._model_name + runner.file_name_suffix_latest, overwrite=True)


    def predict_batch(self, x_batch: Any) -> Tensor:
        """Evaluates the `runner` model's predicted `y_batch` based on an input `x_batch`, without any gradients.

        Args:
            x_batch (Any): Batched data to be passed to the `.forward()` method of the runner's model.

        Returns:
            Tensor: `y_hat_batch`, the output of the forward pass of x_batch.
        """        
        
        with torch.no_grad():
            x_batch = x_batch.to(self._device)
            y_hat = self._model.forward(x_batch)
            return y_hat
    
    
    def quantize_classifier_predictions(batched_predictions: Tensor) -> Tensor:
        """"Snaps" a batch of labels to single values by computing the argmax along axis 1. Used to evaluate the
        accuracy of a classifier model.

        Args:
            batched_predictions (Tensor): A set of predictions (`y_hat_batched`) of the shape `(N,M)`, where `N` is the
            number of samples in the batch and `M` is the number of unique classes that the classifier can predict.

        Returns:
            Tensor: A set of predictions of shape `(N)`, where each item in the output corresponds to the argmax of the
            corresponding row of `batched_predictions`.
        """        
        
        return torch.argmax(batched_predictions, axis=1).cpu()
    
    
    def predict_dataset(self, batch_iterable: Iterable[Tuple[Any, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Calculates a model's predictions for an entire dataset.

        Args:
            batch_iterable (Iterable[Tuple[Any, Tensor]]): an iterable batch data loader which yields tuples of
            batch training data of the form `(x_batch, y_batch)`. The `x_batch` and `y_batch` data for each batch must
            conform to the structure specified in `runner.quantize_classifier_predictions`: `x_batch` must be compatible
            with the model that the `runner` was initialized with, and `y_batch` must be of the shape `(N,M)`, where `N`
            is the number of samples in the batch and `M` is the number of unique classes that the classifier can
            predict.

        Returns:
            Tuple[Tensor, Tensor]: Two tensors of the form `(y, yhat)`, where `y` and `y_hat` are both of shape `(L)`,
            where `L` is the total number of samples in the `batch_iterable`'s dataset and `y_hat` are the predictions
            of `y` using the current model.
        """
        
        y_batches = []
        yhat_batches = []
        
        with tqdm(batch_iterable, desc="Evaluating model predictions", position=0, leave=True, unit="batches") as tqdm_batch_iterable:
            for batched_x, batched_y in iter(tqdm_batch_iterable):
                batched_y = runner.quantize_classifier_predictions(batched_y)
                batched_y_hat = runner.quantize_classifier_predictions(self.predict_batch(batched_x))
                y_batches.append(batched_y)
                yhat_batches.append(batched_y_hat)
        
        y = torch.concat(y_batches, dim=0)
        yhat = torch.concat(yhat_batches, dim=0)
        
        return y, yhat
    
    
    def classifier_accuracy_score(self, batch_iterable: Iterable[Tuple[Any, Tensor]]) -> float:
        """Evaluates the accuracy of the `runner`'s model against some test/validation dataset.

        Args:
            batch_iterable (Iterable[Tuple[Any, Tensor]]): an iterable batch data loader which yields tuples of
            batch training data of the form `(x_batch, y_batch)`. The `x_batch` and `y_batch` data for each batch must
            conform to the structure specified in `runner.quantize_classifier_predictions`: `x_batch` must be compatible
            with the model that the `runner` was initialized with, and `y_batch` must be of the shape `(N,M)`, where `N`
            is the number of samples in the batch and `M` is the number of unique classes that the classifier can
            predict.

        Returns:
            float: The `runner`'s model accuracy against the `batch_iterable` dataset, as a fraction of 1.
        """        
        
        return accuracy_score(*self.predict_dataset(batch_iterable))
    
    
    def plot_model_performance(self, title: str = None, show_loss: bool = False) -> None:
        """Displays a plot of the model's historic accuracy on the train/val datasets, as well as a plot of the model's
        historic training loss, if desired.

        Args:
            title (str, optional): If specified, the title which will be used for the output plot. Defaults to None.
            show_loss (bool, optional): If True, plot the model's historic training loss. Defaults to False.
        """
        
        # Subplots
        if show_loss:
            fig, (ax0, ax1) = plt.subplots(2, sharex=True)
        else:
            ax0 = plt.axes()
        
        # X-axis: make epochs one-indexed
        epoch_range = np.arange(self._epoch) + 1
        
        # Title
        if title:
            plt.suptitle(title)
        
        # ax0: accuracy
        ax0.plot(epoch_range, self._train_acc_history, color="tab:orange", label="Train Accuracy")
        ax0.plot(epoch_range, self._val_acc_history, color="tab:blue", label="Validation Accuracy")
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Accuracy")
        ax0.legend(loc="lower right")
        ax0.grid(which = "both")
        ax0.minorticks_on()
        ax0.tick_params(which="minor", grid_linestyle=":")
        
        # ax1: loss
        if show_loss:
            ax1.plot(epoch_range, self._loss_history, color="tab:orange", label="Train Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend(loc="upper right")
            ax1.grid(which = "both")
            ax1.minorticks_on()
            ax1.tick_params(which="minor", grid_linestyle=":")
            
            # Share the epoch axis
            ax0.label_outer()
            ax1.label_outer()
            plt.subplots_adjust(hspace=0)
            
        
        plt.show()