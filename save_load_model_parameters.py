import os
import torch
import typing

_model_parameters_path = "model_parameters"
_model_parameter_extension = ".pt"
_extension_temp = ".tmp"

def _parameter_file_path(file_name: str) -> str:
    # Add extension, if needed
    if not file_name.endswith(_model_parameter_extension):
        file_name = file_name + _model_parameter_extension
    # Add to parameters folder
    return os.path.join(_model_parameters_path, file_name)


def save_parameters(parameters: dict[str, typing.Any], file_name: str, overwrite: bool = False) -> bool:
    """Saves a dictionary of parameters, such as a state_dict, to a file

    Args:
        parameters (dict[str, Any]): A dictionary of parameters to save
        file_name (str): The name of the file to save the given `parameters` to. NOTE: this should
        be a file name, not a path from the root directory; parameters are automatically saved to
        the `./model_parameters` folder! A file extension of `.pt` is added automatically if not
        included in `file_name`.
        overwrite (bool, optional): If True, overwrites an existing model parameter with a matching
        `file_name`, if it exists. Defaults to False.

    Returns:
        bool: Whether or not the given `parameters` were saved to a file successfully. False only
        if an existing model parameter with a matching `file_name` exists in the
        `./model_parameters` folder and `overwrite` is set to False.
    """
    
    # Get the target file path
    file_path = _parameter_file_path(file_name)

    # Check for an existing file
    file_exists = os.path.isfile(file_path)
    if file_exists and not overwrite:
        # File would be erroneously overwritten, exit early
        return False
    
    if not file_exists:
        # No file exists at the target location, simply save there
        torch.save(parameters, file_path)
    else:
        # A file does exist already at the target location!
        # First, save the parameters to a temp file
        temp_file_path = file_path + _extension_temp
        torch.save(parameters, temp_file_path)
        # Then, delete the old file
        os.remove(file_path)
        # And rename the new one to the same name
        os.rename(temp_file_path, file_path)

    # No errors, return True
    return True


def load_parameters(file_name: str) -> dict[str, typing.Any]:
    """Loads a dictionary of parameters, such as a state_dict, from a file

    Args:
        file_name (str): The name of the file to load parameters. NOTE: this should be a file name,
        not a path from the root directory; parameters are automatically loaded from the
        `./model_parameters` folder! A file extension of `.pt` is added automatically if not
        included in `file_name`.

    Returns:
        dict[str, typing.Any]: A dictionary of parameters which was previously saved to a file; if
        no file with a matching `file_name` exists, returns None.
    """

    # Get the target file path
    file_path = _parameter_file_path(file_name)

    # Check if the file exists; if it doesn't, return early
    if not os.path.exists(file_path):
        return None
    
    # Load in the target parameters
    parameters = torch.load(file_path)
    return parameters

def file_exists(file_name: str) -> bool:
    file_path = _parameter_file_path(file_name)
    return os.path.exists(file_path)