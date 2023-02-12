import neuroscope.templates as templates
import neuroscope.utils as nutils
import neuroscope

from IPython.display import display
from IPython.display import HTML

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
import pandas as pd


website_version = 4
MAKE_META_FILES = True
WEBSITE_DIR = Path(f"/workspace/neuroscope/v{website_version}")
WEBSITE_DIR.mkdir(exist_ok=True)


# %%
def get_num_sub_folders(path):
    path = Path(path)
    return len(list(filter(lambda name: name.is_dir(), path.iterdir())))

def get_list_sub_folders(path):
    path = Path(path)
    return (list(filter(lambda name: name.is_dir(), path.iterdir())))

@lru_cache(maxsize=None)
def get_model_config(model_name):
    return transformer_lens.loading.get_pretrained_model_config(model_name)


def make_random_redirect(neuron):
    return templates.RANDOM_REDIRECT_1D.format(neuron=neuron)

def make_random_redirect_2d(layer, neuron):
    return templates.RANDOM_REDIRECT_1D.format(layer=layer, neuron=neuron)

def test_html_page(html):
    display(HTML(html))
    print(html)

# %%

def gen_main_index_page(model_names):
    """Generates the main index page for the website"""
    columns = ["Model", "Random", "Act Fn", "Dataset", "Layers", "Neurons per Layer", "Total Neurons", "Params"]
    rows = []
    for name in model_names:
        cfg = get_model_config(name)
        rows.append([
            f"<a href='./{name}/index.html'>{name}</a>",
            f"<a href='./{name}/random.html'>Random</a>",
            cfg.act_fn[:4],
            nutils.model_name_to_fancy_data_name(name),
            cfg.n_layers,
            cfg.d_mlp,
            cfg.d_mlp * cfg.n_layers,
            cfg.n_params,
        ])
    df = pd.DataFrame(rows, columns=columns)

    add_commas = lambda x: f"{x:,}"

    return templates.NEUROSCOPE_MAIN_INDEX.format(
        models_table = df.to_html(index=False, escape=False, justify="right", formatters={"Total Neurons":add_commas, "Params":add_commas, "Neurons per Layer":add_commas}),
    )


def gen_model_page(model_name):
    """Generates the index page for a single model. Fills out templates.MODEL_INDEX"""
    
    cfg = get_model_config(model_name)
    
    rows = []
    for l in range(cfg.n_layers):
        rows.append(
            [
                f"<b>Layer #{l}</b>",
                f"<a href='./{l}/0.html'>First</a>",
                f"<a href='./{l}/random.html'>Random</a>",
                f"<a href='./{l}/{cfg.d_mlp - 1}.html'>Last</a>",
            ]
        )
    df = pd.DataFrame(rows, columns=["Layer", "First", "Random", "Last"])
    
    return templates.MODEL_INDEX.format(
        max_layer = cfg.n_layers - 1,
        max_neuron = cfg.d_mlp - 1,
        fancy_model_name = nutils.get_fancy_model_name(model_name),
        fancy_data_name = nutils.model_name_to_fancy_data_name(model_name),
        model_name = model_name,
        model_layers_table = df.to_html(escape=False, header=False, index=False, justify="right"),
    )
def make_page_file(path, html):
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_text(html)

# %%
if __name__=="__main__":
    test_html_page(gen_model_page("solu-12l"))
    test_html_page(gen_main_index_page(["gpt2-xl", "solu-12l", "solu-1l", "solu-2l"]))



# %%
if __name__=="__main__" and MAKE_META_FILES:
    model_names = list(map(lambda k: k.name, get_list_sub_folders(WEBSITE_DIR)))
    model_names.sort(key = lambda name: get_model_config(name).n_params)
    model_names.sort(key = lambda name: nutils.model_name_to_data_name(name))
    print(model_names)
    test_html_page(gen_main_index_page(model_names))
else:
    model_names = []
# %%
if __name__=="__main__" and MAKE_META_FILES:
    print("Index:")
    index_html = gen_main_index_page(model_names)
    test_html_page(index_html)
    make_page_file((WEBSITE_DIR / "index.html"), (index_html))
    # (WEBSITE_DIR / "index.html").write_text(index_html)
    for name in model_names:
        print("Starting model", name)
        cfg = get_model_config(name)
        print("Total Layers:", cfg.n_layers)
        print("Num Neurons:", cfg.d_mlp)
        print(f"Model: {name}")
        model_html = gen_model_page(name)
        test_html_page(model_html)
        folder = WEBSITE_DIR / name
        assert folder.exists()
        
        make_page_file((folder / "index.html"), (model_html))
        
        make_page_file((folder / "model.html"), (templates.REDIRECT_TO_INDEX))
        
        make_page_file((folder / "random.html"), (make_random_redirect_2d(cfg.n_layers, cfg.d_mlp)))
        
        for subfolder in get_list_sub_folders(folder):
            make_page_file((subfolder / "random.html"), (make_random_redirect(cfg.d_mlp)))
            make_page_file((subfolder/"index.html"), (templates.REDIRECT_TO_INDEX_ONE_UP))
            
# %%
