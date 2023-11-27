import torch
from torch import nn


class review_LSTM(nn.Module):
    """An LSTM implementation where one label prediction is made per embedded review
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_classifier: nn.Module, num_layers: int = 1):
        """Creates a review LSTM with the specified parameters.

        Args:
            input_size (int): The number of input features to the LSTM module, equal to the number of embedded features
            per token of a review.
            hidden_size (int): The number of hidden features in the LSTM.
            output_classifier (nn.Module): A classifier which, given an input of shape `(N, [hidden_size*num_layers])`,
            returns a tensor of shape `(N*num_classes)`, where `N` is the input batch size, `hidden_size` is passed as
            the above parameter, and `num_classes` is the number of unique classes the model predicts (typically 2 in
            our case, either positive or negative). For classification tasks, the `output_classifier` should contain a
            softmax output layer (`dim=1`).
        """
        
        # Initialize the module
        super().__init__()
        
        # Store parameters
        self._LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self._hidden_size = hidden_size
        self._output_classifier = output_classifier
        self._num_layers = num_layers
    
    def forward(self, x: torch.Tensor | nn.utils.rnn.PackedSequence) -> torch.Tensor:
        """Forward pass of the LSTM module and output classifier.

        Args:
            x (torch.Tensor | nn.utils.rnn.PackedSequence): A set of input features. For unbatched inputs, a tensor of
            shape `(L*input_size)` should be given; for batched inputs, a tensor of shape `(N*L*input_size)` should be
            given; a PackedSequence object may also be passed for either batched or unbatched inputs. Here, `N` is the
            number of samples in a batch, `L` is the number of tokens in the input (PackedSequence objects allow this
            to be different between samples in a batched input), and `input_size` is the value defined in the
            `review_LSTM`'s initialization.

        Returns:
            torch.Tensor: A tensor of shape `(N*num_classes)`, where `N` is the number of samples in the batched input
            (possibly 1) and `num_classes` is the is the number of unique classes the model predicts; the output shape
            is determined by the `output_classifier` with which the `review_LSTM` was initialized.
        """
        
        # Give the review data to the LSTM to munch on
        output, (h_n, c_n) = self._LSTM.forward(x)
        
        # Based on the final hidden state, use the output classifier to make a label prediction
        c_n = c_n.reshape((-1, self._hidden_size * self._num_layers))
        yhat = self._output_classifier.forward(c_n)
        
        # Return the results from the output classifier
        return yhat