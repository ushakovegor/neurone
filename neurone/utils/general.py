import os
import yaml
import shutil as sh


def makedir_overwrite(dir_path, overwrite):
    """Creates dir with specified path

    Parameters
    ----------
    dir_path: str
        path to dir to create
    overwrite: bool
        Specifies the behaviour for the case when the directory with dir_path exists
        If True, the dir will be deleted. If False, the dir will be left intact and the Exceptrion will be raised.
    """
    if os.path.exists(dir_path):
        if overwrite:
            sh.rmtree(dir_path)
        else:
            raise Exception(
                "The directory exists and overwrite mode is disabled: \n %s \n Aborting."
                % dir_path
            )
    os.makedirs(dir_path, exist_ok=True)


def load_yaml(path):
    """
    Safely loads contents from yaml file.

    Parameters
    ----------
    path: str
        path to yaml

    Returns
    -------
    contents: dict or list
        contents of yaml file
    """
    with open(path, "r") as file:
        return yaml.safe_load(file)


def write_yaml(path, contents, overwrite=False):
    """
    Safely writes contents to the yaml file.

    Parameters
    ----------
    path: str
        path to yaml
    contents: dict or list
        contetns to write
    overwrite: bool
        Defines the behavior in case of already present file. If true, the file will be overwritten, if false, the exeption is raised.
    """

    if os.path.exists(path):
        raise FileExistsError("File %s exists and overwrite mode is disabled.")

    with open(path, "w+") as file:
        yaml.safe_dump(contents, file)
