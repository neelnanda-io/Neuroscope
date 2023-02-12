# %%
import einops
import re
import os
from pathlib import Path
import huggingface_hub
import torch
import json
import numpy as np
import plotly.express as px
import logging
import shutil
import argparse
import datasets
from typing import Tuple, Union
from IPython import get_ipython
from transformer_lens.utils import get_corner
from functools import lru_cache
import transformer_lens

CACHE_DIR = Path.home() / ("cache")
REPO_ROOT = Path.home() / ("hf_repos/")
OLD_CHECKPOINT_DIR = Path.home() / ("solu_project/solu_checkpoints/")
CHECKPOINT_DIR = Path.home() / ("solu_project/saved_models/")


def download_file_from_hf(repo_name, file_name, subfolder=".", cache_dir=CACHE_DIR):
    file_path = huggingface_hub.hf_hub_download(
        repo_id=f"NeelNanda/{repo_name}",
        filename=file_name,
        subfolder=subfolder,
        cache_dir=cache_dir,
    )
    print(f"Saved at file_path: {file_path}")
    if file_path.endswith(".pth"):
        return torch.load(file_path)
    elif file_path.endswith(".json"):
        return json.load(open(file_path, "r"))
    else:
        print("File type not supported:", file_path.split(".")[-1])
        return file_path


# %%



# %%
def push_to_hub(repo_dir):
    """Pushes a directory/repo to HuggingFace Hub

    Args:
        repo_dir (str or Repository): The directory of the relevant repo
    """
    if isinstance(repo_dir, huggingface_hub.Repository):
        repo_dir = repo_dir.repo_dir
    # -C means "run command as though you were in that directory"
    # Uses explicit git calls on CLI which is way faster than HuggingFace's Python API for some reason
    os.system(f"git -C {repo_dir} add .")
    os.system(f"git -C {repo_dir} commit -m 'Auto Commit'")
    os.system(f"git -C {repo_dir} push")

def upload_folder_to_hf(folder_path, repo_name=None, debug=False):
    folder_path = Path(folder_path)
    if repo_name is None:
        repo_name = folder_path.name
    repo_folder = folder_path.parent / (folder_path.name + "_repo")
    repo_url = huggingface_hub.create_repo(repo_name, exist_ok=True)
    repo = huggingface_hub.Repository(str(repo_folder), repo_url)

    for file in folder_path.iterdir():
        if debug:
            print(file.name)
        file.rename(repo_folder / file.name)
    push_to_hub(repo.local_dir)



# %%
def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.

    If in Ipython, just returns with no changes
    """
    if get_ipython() is not None:
        # Is in IPython
        print("In IPython - skipped argparse")
        return default_cfg
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg


# %%
class TokenDatasetWrapper:
    """
    A wrapper around a HuggingFace Dataset which allows the slicing syntax (ie dataset[4], dataset[4:8], dataset[[5, 1, 7, 8]] etc.)
    Used to allow for uint16 datasets which are Torch incompatible (used for the Pile), but consume half the space!.
    Explicitly used for datasets of tokens
    """

    def __init__(self, dataset):
        if isinstance(dataset, datasets.Dataset):
            self.dataset = dataset
        elif isinstance(dataset, TokenDatasetWrapper):
            self.dataset = dataset.dataset
        elif isinstance(dataset, datasets.DatasetDict):
            self.dataset = dataset["train"]
        else:
            raise ValueError(f"Invalid dataset type: {type(dataset)}")

        self.is_unint16 = self.dataset.features["tokens"].feature.dtype == "uint16"
        if self.is_unint16:
            self.dataset = self.dataset.with_format("numpy")
        else:
            self.dataset = self.dataset.with_format("torch")
        
    def __getitem__(self, idx) -> torch.Tensor:
        tokens = self.dataset[idx]['tokens']
        if self.is_unint16:
            tokens = tokens.astype(np.int32)
            return torch.tensor(tokens)
        else:
            return tokens
    
    def __len__(self):
        return len(self.dataset)

@lru_cache(maxsize=None)
def get_dataset(dataset_name: str, local=False) -> TokenDatasetWrapper:
    """Loads in one of the model datasets over which we take the max act examples. If local, loads from local folder, otherwise loads from HuggingFace Hub

    Args:
        dataset_name (str): Name of the dataset, must be one of the entries in the dictionary
        local (bool, optional): Whether to load from a local folder or remotely. Defaults to False.

    Returns:
        datasets.Dataset: _description_
    
    Test:
        for name in ["c4", "code", "pile", "openwebtext", "pile-big", "c4-code"]:
            code_remote = nutils.get_dataset(name, local=False)
            code_local = nutils.get_dataset(name, local=True)
            a = torch.randint(0, len(code_remote), (100,))
            rtokens = code_remote[a]
            ltokens = code_local[a]
            try:
                assert len(code_remote) == len(code_local)
                assert (rtokens==ltokens).all()
                print("Success", name)
            except:
                print("Failed", name)
    """
    if local:
        local_dataset_names = {
            "c4-code": "c4_code_valid_tokens.hf",
            "c4": "c4_valid_tokens.hf",
            "code": "code_valid_tokens.hf",
            "pile": "pile_big_int32.hf",
            "pile-big": "pile_big_int32.hf",
            "pile-big-uint16": "pile_big_int16.hf",
            "openwebtext": "openwebtext_tokens.hf",
        }
        tokens = datasets.load_from_disk("/workspace/data/" + local_dataset_names[dataset_name])
        tokens = tokens.with_format("torch")
        return TokenDatasetWrapper(tokens)
    else:
        remote_dataset_names = {
            "c4": "NeelNanda/c4-tokenized-2b",
            "code": "NeelNanda/code-tokenized",
            "pile": "NeelNanda/pile-small-tokenized-2b",
            "pile-small": "NeelNanda/pile-small-tokenized-2b",
            "pile-big": "NeelNanda/pile-tokenized-10b",
            "pile-big-uint16": "NeelNanda/pile-tokenized-10b",
            "openwebtext": "NeelNanda/openwebtext-tokenized-9b",
        }
        if dataset_name=="c4-code":
            c4_data = datasets.load_dataset(remote_dataset_names["c4"], split="train")
            code_data = datasets.load_dataset(remote_dataset_names["code"], split="train")
            tokens = datasets.concatenate_datasets([c4_data, code_data])
        else:
            tokens = datasets.load_dataset(remote_dataset_names[dataset_name], split="train")
        tokens = tokens.with_format("torch")
        return TokenDatasetWrapper(tokens)



class MaxStore:
    """Used to calculate max activating dataset examples - takes in batches of activations repeatedly, and tracks the top_k examples activations + indexes"""

    def __init__(self, top_k, length, device="cuda"):
        self.top_k = top_k
        self.length = length
        self.device = device

        self.max = -torch.inf * torch.ones(
            (top_k, length), dtype=torch.float32, device=device
        )
        self.index = -torch.ones((top_k, length), dtype=torch.long, device=device)

        self.counter = 0
        self.total_updates = 0
        self.batches_seen = 0

    def update(self, new_act, new_index):
        min_max_act, min_indices = self.max.min(dim=0)
        mask = min_max_act < new_act
        num_updates = mask.sum().item()
        self.max[min_indices[mask], mask] = new_act[mask]
        self.index[min_indices[mask], mask] = new_index[mask]
        self.total_updates += num_updates
        return num_updates

    def batch_update(self, activations, text_indices=None):
        """
        activations: Shape [batch, length]
        text_indices: Shape [batch,]

        activations is the largest MLP activation, text_indices is the index of the text strings.

        Sorts the activations into descending order, then updates with each column until we stop needing to update
        """
        batch_size = activations.size(0)
        new_acts, sorted_indices = activations.sort(0, descending=True)
        if text_indices is None:
            text_indices = torch.arange(
                self.counter,
                self.counter + batch_size,
                device=self.device,
                dtype=torch.int64,
            )
        new_indices = text_indices[sorted_indices]
        for i in range(batch_size):
            num_updates = self.update(new_acts[i], new_indices[i])
            if num_updates == 0:
                break
        self.counter += batch_size
        self.batches_seen += 1

    def save(self, dir, folder_name=None):
        if folder_name is not None:
            path = dir / folder_name
        else:
            path = dir
        path.mkdir(exist_ok=True)
        torch.save(self.max, path / "max.pth")
        torch.save(self.index, path / "index.pth")
        with open(path / "config.json", "w") as f:
            filt_dict = {
                k: v for k, v in self.__dict__.items() if k not in ["max", "index"]
            }
            json.dump(filt_dict, f)
        print("Saved Max Store to:", path)

    def switch_to_inference(self):
        """Switch from updating mode to inference - move to the CPU and sort by max act."""
        self.max = self.max.cpu()
        self.index = self.index.cpu()
        self.max, indices = self.max.sort(dim=0, descending=True)
        self.index = self.index.gather(0, indices)

    @classmethod
    def load(cls, dir, folder_name=None, continue_updating=False, transpose=False):
        dir = Path(dir)
        if folder_name is not None:
            path = dir / folder_name
        else:
            path = dir

        max = torch.load(path / "max.pth")
        index = torch.load(path / "index.pth")
        if transpose:
            max = max.T
            index = index.T
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        mas = cls(config["top_k"], config["length"])
        for k, v in config.items():
            mas.__dict__[k] = v
        mas.max = max
        mas.index = index
        if not continue_updating:
            mas.switch_to_inference()
        return mas

    def __repr__(self):
        return f"MaxStore(top_k={self.top_k}, length={self.length}, counter={self.counter}, total_updates={self.total_updates}, device={self.device})\n Max Values: {get_corner(self.max)}\n Indices: {get_corner(self.index)}"

# %%

def model_name_to_data_name(model_name):
    if "old" in model_name or "pile" in model_name:
        data_name = "pile"
    elif "pythia" in model_name:
        data_name = "pile-big"
    elif "gpt2" in model_name:
        data_name = "openwebtext"
    elif model_name.startswith("solu") or model_name.startswith("gelu") or model_name.startswith("attn-only"):
        # Note that solu-{}l-pile will go into the first set!
        data_name = "c4-code"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return data_name

def model_name_to_fancy_data_name(model_name):
    fancy_data_names = {
        "c4-code": "80% C4 (Web Text) and 20% Python Code",
        "c4": "C4 (Web Text)",
        "code": "Python Code",
        "pile": "The Pile",
        "pile-big": "The Pile",
        "pile-small": "The Pile",
        "openwebtext": "Open Web Text",
    }
    data_name = model_name_to_data_name(model_name)
    return fancy_data_names[data_name]

def get_fancy_model_name(model_name):
    cfg = transformer_lens.loading.get_pretrained_model_config(model_name)
    if cfg.act_fn in ["solu", "solu_ln"]:
        return f"SoLU Model: {cfg.n_layers} Layers, {cfg.d_mlp} Neurons per Layer"
    elif "gelu" in cfg.act_fn:
        if cfg.original_architecture == "neel":
            return f"GELU Model: {cfg.n_layers} Layers, {cfg.d_mlp} Neurons per Layer"
        elif cfg.original_architecture == "GPT2LMHeadModel":
            return f"GPT-2 {model_name.split('-')[-1].capitalize()}: {cfg.n_layers} Layers, {cfg.d_mlp} Neurons per Layer"
    else:
        raise ValueError(f"{model_name} Invalid Model Name for fancy model name")
    