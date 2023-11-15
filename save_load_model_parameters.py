import os
import torch

_model_parameters_path = "model_parameters"
_model_parameter_extension = ".pt"

def _parameter_file_path(file_name: str) -> str:
    # Add extension, if needed
    if not file_name.endswith(_model_parameter_extension):
        file_name = file_name + _model_parameter_extension
    # Add to parameters folder
    return os.path.join(_model_parameters_path, file_name)


def save_parameters(module_or_optimizer: torch.nn.Module | torch.optim.Optimizer, file_name: str, overwrite: bool = False) -> bool:
    """Saves the parameters of a `torch.nn.Module` or `torch.optim.Optimizer` to a file for later
    recall

    Args:
        module_or_optimizer (torch.nn.Module | torch.optim.Optimizer): A module or optimizer whose
        parameters will be saved; saveable parameters are listed by the module or optimizer's
        `.state_dict()` method
        file_name (str): The name of the file to save the model/optimizer parameters to. **NOTE**:
        this should be a file name, not a path from the root directory; parameters are
        automatically saved to the `./model_parameters` folder! A file extension of `.pt` is added
        automatically if not included in `file_name`. 
        overwrite (bool, optional): If True, overwrites an existing model parameter with a matching
        `file_name`, if it exists. Defaults to False.

    Returns:
        bool: Whether or not the module or optimizer's parameters were saved to a file
        successfully. False only if an existing model parameter with a matching `file_name` exists
        and `overwrite` is set to False.
    """  
    
    # Get the target file path
    file_path = _parameter_file_path(file_name)

    # Check for an existing file and return early if it would be erroneously overwritten
    if (not overwrite) and os.path.exists(file_path):
        return False
    
    # Write the parameter to the target file
    torch.save(module_or_optimizer.state_dict(), file_path)

    # No errors, return True
    return True


def load_parameters(module_or_optimizer: torch.nn.Module | torch.optim.Optimizer, file_name: str) -> bool:
    """Loads the parameters of a `torch.nn.Module` or `torch.optim.Optimizer` from a file which
    was previously created by `save_parameters()`

    Args:
        module_or_optimizer (torch.nn.Module | torch.optim.Optimizer): A module or optimizer whose
        parameters will be restored from the save file; saveable parameters are listed by the
        module or optimizer's `.state_dict()` method
        file_name (str): The name of the file to load the model/optimizer parameters from.
        **NOTE**: this should be a file name, not a path from the root directory; parameters are
        automatically loaded from the `./model_parameters` folder! A file extension of `.pt` is
        added automatically if not included in `file_name`. 

    Returns:
        bool: Whether or not the module or optimizer's parameters were loaded from a file
        successfully. False only if no existing model parameter with a matching `file_name` was
        found (likely because `save_parameters()` hasn't been called yet)
    """ 

    # Get the target file path
    file_path = _parameter_file_path(file_name)

    # Check if the file exists; if it doesn't, return early
    if not os.path.exists(file_path):
        return False
    
    # Load in the module/optimizer parameters
    module_or_optimizer_state = torch.load(file_path)
    module_or_optimizer.load_state_dict(module_or_optimizer_state)
    
    # No errors, return True
    return True