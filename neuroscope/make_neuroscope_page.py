# %%
# Setup
from neel.imports import *
# from solu.microscope.microscope import *

pio.renderers.default = "vscode"
torch.set_grad_enabled(False)
import gradio as gr
from transformer_lens import HookedTransformer
from transformer_lens.utils import to_numpy
from IPython.display import HTML
import argparse
import solu.utils as sutils

import circuitsvis as cv
from circuitsvis.utils.render import RenderedHTML, render, render_cdn
# %%
# Variables
HACKY_VERSION_FIX = True

debug = IN_IPYTHON
DEFAULT_CFG = dict(
    layer=0,
    model_name="solu-2l",
    # data_name = "c4-code",
    version=2,
    top_k=20,
    top_logits=10,
    bottom_logits=5,
    website_version=4,
    debug=False,
    use_logits=False,
    truncated_prefix_length=50,
    truncated_suffix_length=10,
)
if not IN_IPYTHON:
    cfg = sutils.arg_parse_update_cfg(DEFAULT_CFG)
else:
    cfg = DEFAULT_CFG
layer = cfg["layer"]
model_name = cfg["model_name"]
# data_name = cfg['data_name']
version = cfg["version"]
if HACKY_VERSION_FIX and version==2 and (model_name in [f"{act_fn}-{n}l" for act_fn in ['solu', 'gelu'] for n in range(1, 5)] or model_name.endswith("pile")):
    print(f"Hackily fixing version for {model_name}")
    version = 1
top_k = cfg["top_k"]
top_logits = cfg["top_logits"]
bottom_logits = cfg["bottom_logits"]
website_version = cfg["website_version"]
debug = cfg["debug"] or IN_IPYTHON
use_logits = cfg["use_logits"]
truncated_prefix_length = cfg["truncated_prefix_length"]
truncated_suffix_length = cfg["truncated_suffix_length"]

def model_name_to_data_name(model_name):
    if "old" in model_name or "pile" in model_name:
        data_name = "pile"
    elif "pythia" in model_name:
        data_name = "pile-big"
    elif "gpt" in model_name:
        data_name = "openwebtext"
    elif model_name.startswith("solu") or model_name.startswith("gelu"):
        data_name = "c4-code"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return data_name
data_name = model_name_to_data_name(model_name)
print("Data name:", data_name)


WEBSITE_DIR = Path(
    f"/workspace/neuroscope{'/debug' if debug else ''}/v{website_version}/{model_name}/{layer}"
)
WEBSITE_DIR.mkdir(parents=True, exist_ok=True)
fancy_data_names = {
    "c4-code": "80% C4 (Web Text) and 20% Python Code",
    "c4": "C4 (Web Text)",
    "code": "Python Code",
    "pile": "The Pile",
    "pile-big": "The Pile",
    "openwebtext": "Open Web Text",
}
# fancy_model_names = {
#     "solu-1l": "SoLU Model: 1 Layers, 512 Neurons per Layer",
#     "solu-2l": "SoLU Model: 2 Layers, 512 Neurons per Layer",
#     "solu-3l": "SoLU Model: 3 Layers, 512 Neurons per Layer",
#     "solu-4l": "SoLU Model: 4 Layers, 512 Neurons per Layer",
#     "solu-6l": "SoLU Model: 6 Layers, 768 Neurons per Layer",
#     "solu-8l": "SoLU Model: 8 Layers, 512 Neurons per Layer",
#     "solu-10l": "SoLU Model: 10 Layers, 512 Neurons per Layer",
#     "solu-12l": "SoLU Model: 12 Layers, 512 Neurons per Layer",
#     "solu-1l-old": "SoLU Model: 1 Layers, 1024 Neurons per Layer",
#     "solu-2l-old": "SoLU Model: 2 Layers, 736 Neurons per Layer",
#     "solu-4l-old": "SoLU Model: 4 Layers, 512 Neurons per Layer",
#     "solu-6l-old": "SoLU Model: 6 Layers, 768 Neurons per Layer",
#     "solu-8l-old": "SoLU Model: 8 Layers, 1024 Neurons per Layer",
#     "solu-10l-old": "SoLU Model: 10 Layers, 1280 Neurons per Layer",
# }

def get_fancy_model_name(model_name):
    cfg = loading.get_pretrained_model_config(model_name)
    if cfg.act_fn in ["solu", "solu_ln"]:
        return f"SoLU Model: {cfg.n_layers} Layers, {cfg.d_mlp} Neurons per Layer"
    elif "gelu" in cfg.act_fn:
        if cfg.original_architecture == "neel":
            return f"GELU Model: {cfg.n_layers} Layers, {cfg.d_mlp} Neurons per Layer"
        elif cfg.original_architecture == "GPT2LMHeadModel":
            return f"GPT-2 {model_name.split('-')[-1].capitalize()}: {cfg.n_layers} Layers, {cfg.d_mlp} Neurons per Layer"
    else:
        raise ValueError(f"{model_name} Invalid Model Name for fancy model name")
# %%
# Loading
model = HookedTransformer.from_pretrained(model_name)

# %%
if data_name == "c4-code":
    c4_data = sutils.get_dataset("c4")
    code_data = sutils.get_dataset("code")
    dataset_offset = len(c4_data)
    data = datasets.concatenate_datasets([c4_data, code_data])
    
    code_store = sutils.MaxStore.load(f"/workspace/solu_outputs/neuron_max_act/code/{model_name}/v{version}/{layer}")
    c4_store = sutils.MaxStore.load(f"/workspace/solu_outputs/neuron_max_act/c4/{model_name}/v{version}/{layer}")
    
    cat_max = torch.cat([c4_store.max, code_store.max], dim=0)
    cat_index = torch.cat([c4_store.index, code_store.index + dataset_offset], dim=0)
    
    sorted_cat_max, indices = cat_max.sort(dim=0, descending=True)
    sorted_cat_index = cat_index.gather(0, indices)
    
    store = sutils.MaxStore(c4_store.top_k, c4_store.length, device="cpu")
    store.max = sorted_cat_max[:store.top_k]
    store.index = sorted_cat_index[:store.top_k]
elif data_name == "pile":
    data = sutils.get_dataset(data_name)
    store = sutils.MaxStore.load(f"/workspace/solu_outputs/neuron_max_act/{data_name}/{model_name}/v{version}/{layer}")
elif data_name == "pile-big":
    data = sutils.get_dataset(data_name)
    store = sutils.MaxStore.load(f"/workspace/solu_outputs/neuron_max_act/pile/{model_name}/v{version}/{layer}")
elif data_name == "openwebtext":
    data = sutils.get_dataset(data_name)
    store = sutils.MaxStore.load(f"/workspace/solu_outputs/neuron_max_act/{data_name}/{model_name}/v{version}/{layer}")
else:
    raise ValueError(f"Invalid data name {data_name}")

# %%
W_U = model.W_U
# if not isinstance(model.ln_final, transformer_lens.components.LayerNormPre):
#     print("Folding in Layer Norm")
#     W_U = model.ln_final.w[:, None] * W_U
W_logit = model.blocks[layer].mlp.W_out @ W_U
print("W_logit:", W_logit.shape)

# %%
# def get_neuron_acts(text, neuron_index):
#     """Hacky way to get out state from a single hook - we have a single element list and edit that list within the hook."""
#     cache = {}

#     def caching_hook(act, hook):
#         cache["activation"] = act[0, :, neuron_index]

#     model.run_with_hooks(
#         text,
#         fwd_hooks=[(f"blocks.{layer}.mlp.hook_mid", caching_hook)],
#         return_type=None,
#     )
#     return to_numpy(cache["activation"])


def get_batch_neuron_acts(tokens, neuron_index):
    """Hacky way to get out state from a single hook - we have a single element dict and edit that dict within the hook.

    We feed in a batch x pos batch of tokens, and get out a batch x pos tensor of activations.
    """
    cache = {}

    def caching_hook(act, hook):
        cache["activation"] = act[:, :, neuron_index].to(torch.float32)

    # Data already comes with bos prepended
    with torch.autocast("cuda", torch.bfloat16):
        if model.cfg.act_fn == "solu_ln" or model.cfg.act_fn == "solu":
            model.run_with_hooks(
                tokens,
                fwd_hooks=[(f"blocks.{layer}.mlp.hook_mid", caching_hook)],
                return_type=None,
                stop_at_layer=layer+1,
            )
        else:
            model.run_with_hooks(
                tokens,
                fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", caching_hook)],
                return_type=None,
                stop_at_layer=layer+1,
            )


    return cache["activation"].cpu()

def array_to_trunc_floats(array: np.ndarray, decimal_places: int = 6):
    if len(array.shape)==0:
        return array.item()
    elif len(array.shape)==1:
        return [round(float(i), decimal_places) for i in array]
    elif len(array.shape)==2:
        return [[round(float(i), decimal_places) for i in subarray] for subarray in array]
    elif len(array.shape)==3:
        return [[[round(float(i), decimal_places) for i in subsubarray] for subsubarray in subarray] for subarray in array]
    else:
        raise ValueError(f"Invalid Array shape {array.shape}")



# Test
# For some reason, there's slight differences in the activations, but doesn't matter lol. Confusing though! Also not in a consistent direction. I wonder if it's downstream of how the tensor is stored or smth?
if debug:
    out = get_batch_neuron_acts(data[store.index[:, 5]]["tokens"], 5)
    print(out.shape)
    print(out.max(1).values)
    print(store.max[:, 5])
    print(
        torch.isclose(
            out.max(1).values, store.max[:, 5], rtol=1e-3, atol=1e-5
        )
    )
# %%
# Make HTML
# This is some CSS (tells us what style )to give each token a thin gray border, to make it easy to see token separation
grey_color = 180
# style_string = f"""<style> 
#     span.token {{
#         border: 1px solid rgb({grey_color}, {grey_color}, {grey_color});
#         white-space: pre;
#         color: rgb(0, 0, 0);
#         }} 
#     div.token-text {{
#         word-wrap: normal;
#         }} 
#     </style>"""
#  display: flex; flex-wrap: wrap;
# if debug:
#     print(style_string)
# if debug and IN_IPYTHON:
#     display(
#         HTML(
#             style_string
#             + "<span class='token'>Text!</span><span class='token'>Tixt</span>"
#         )
#     )
# %%
def make_colored_tokens(tokens, values, min_value, max_value) -> str:
    return render_cdn(
        "ColoredTokens",
        tokens=tokens,
        values=values,
        minValue=float(min_value),
        maxValue=float(max_value),
    )

def calculate_color(val, max_val, min_val):
    # Hacky code that takes in a value val in range [min_val, max_val], normalizes it to [0, 1] and returns a color which interpolates between slightly off-white and red (0 = white, 1 = red)
    # We return a string of the form "rgb(240, 240, 240)" which is a color CSS knows
    normalized_val = (val - min_val) / max_val
    return f"rgb(250, {round(250*(1-normalized_val))}, {round(250*(1-normalized_val))})"


def make_single_token_text(str_token, act, max_val, min_val):
    return f"<span class='token' style='background-color:{calculate_color(act, max_val, min_val)}' >{str_token}</span>"


def make_header(neuron_index):
    htmls = []
    htmls.append(f"<div style='font-size:medium;'>")
    if neuron_index > 0:
        htmls.append(f"< <a href='./{neuron_index-1}.html'>Prev</a> | ")
    htmls.append(f"<a href='../../index.html'>Home</a> | ")
    htmls.append(f"<a href='../model.html'>Model</a> | ")
    htmls.append(
        f"<a href='javascript:void(0)' onclick=\"location.href='./' + Math.floor(Math.random() * {model.cfg.d_mlp}) + '.html'\">Random</a> | "
        # f"<a href='./{random.randint(0, model.cfg.d_mlp-1)}.html'>Random</a> | "
    )
    if neuron_index < model.cfg.d_mlp:
        htmls.append(f"<a href='./{neuron_index+1}.html'>Next</a> >")
    htmls.append(f"</div>")
    htmls.append(f"<h1>Model: {get_fancy_model_name(model_name)}</h1>")
    htmls.append(f"<h1>Dataset: {fancy_data_names[data_name]}</h1>")
    htmls.append(f"<h2>Neuron {neuron_index} in Layer {layer} </h2>")
    htmls.append(
        f"<h2>Load this data into an <a href='https://neelnanda.io/interactive-neuroscope'>Interactive Neuroscope</a></h2>"
    )
    htmls.append(
        f"<h3><a href='https://neelnanda.io/neuroscope-docs'>See Documentation here</a></h3>"
    )
    htmls.append(
        f"<h3>Transformer Lens Loading: <span style='font-family: \"Courier New\"'>HookedTransformer.from_pretrained('{model_name}')</span></h3>"
    )
    return "\n".join(htmls)


def make_logits(neuron_index):
    if not use_logits:
        return ""
    htmls = []
    htmls.append("<h3>Direct Logit Effect</h3>")
    logit_vec, logit_indices = W_logit[neuron_index].sort(descending=True)
    for i in range(top_logits):
        htmls.append(
            f"<p style='color: blue; font-family: \"Courier New\"'>#{i} +{logit_vec[i].item():.4f} <span class='token'>{model.to_string([logit_indices[i].item()])}</span></p>"
        )
    htmls.append("<p>...</p>")
    for i in range(bottom_logits):
        htmls.append(
            f"<p style='color: red; font-family: \"Courier New\"'>#{i} {logit_vec[-(i+1)].item():.4f} <span class='token'>{model.to_string([logit_indices[-(i+1)].item()])}</span></p>"
        )
    return "\n".join(htmls)


def make_token_text(
    tokens: np.ndarray, acts: np.ndarray, max_val: float, min_val: float, index: int, data_index: int
):
    htmls = []
    htmls.append(
        f"<h4>Max Range: <b>{max_val:.4f}</b>. Min Range: <b>{min_val:.4f}</b></h4>"
    )
    
    act_max = acts.max()
    act_min = acts.min()
    htmls.append(
        f"<h4>Max Act: <b>{act_max:.4f}</b>. Min Act: <b>{act_min:.4f}</b></h4>"
    )

    if data_name == "c4-code":
        current_fancy_data_name = fancy_data_names['c4' if data_index < len(c4_data) else 'code']
    else:
        current_fancy_data_name = fancy_data_names[data_name]
    htmls.append(
        f"<h4>Data Index: <b>{data_index}</b> ({current_fancy_data_name})</h4>"
    )

    max_token_index = int(acts.argmax())
    htmls.append(
        f"<h4>Max Activating Token Index: <b>{max_token_index}</b></h4>"
    )
    
    htmls.append("<details><summary><b style='color: red'>Click toggle to see full text</b>")
    htmls.append("<h3>Truncated</h3>")
    trunc_indices = list(range(
        max(0, max_token_index - truncated_prefix_length),
        # + 1 so that truncated_suffix_length doesn't delete the max act token!
        min(len(tokens), max_token_index + truncated_suffix_length + 1),
    ))

    
    trunc_str_tokens = model.to_str_tokens(tokens[trunc_indices])
    trunc_acts = acts[trunc_indices]
    tokens_render = make_colored_tokens(
        tokens=trunc_str_tokens,
        values=array_to_trunc_floats(trunc_acts),
        min_value=min_val,
        max_value=max_val,
    )
    htmls.append(tokens_render)
    htmls.append(f"</summary>")
    htmls.append(f"<h3>Full Text #{index}</h3>")
    str_tokens = model.to_str_tokens(tokens)
    tokens_render = make_colored_tokens(
        tokens=str_tokens,
        values=array_to_trunc_floats(acts),
        min_value=min_val,
        max_value=max_val,
    )
    htmls.append(tokens_render)
    htmls.append("</details>")
    return "\n".join(htmls)


def make_token_texts(tokens: np.ndarray, acts: np.ndarray, neuron_index: int, data_indices: torch.Tensor):
    max_val = float(acts.max())
    min_val = -max_val
    return "\n<hr>\n".join(
        [
            f"<h2>Text #{i}</h2>"
            + make_token_text(tokens[i], acts[i], max_val, min_val, index=i, data_index=data_indices[i].item())
            for i in range(top_k)
        ]
    )


def make_html(neuron_index):
    data_indices = store.index[:, neuron_index]
    tokens = data[data_indices]["tokens"]
    acts = get_batch_neuron_acts(tokens, neuron_index)
    acts = to_numpy(acts)
    tokens = to_numpy(tokens)
    htmls = [
        make_header(neuron_index),
        make_logits(neuron_index),
        make_token_texts(tokens, acts, neuron_index, data_indices),
    ]
    return "\n<hr>\n".join(htmls)

if debug and IN_IPYTHON:
    test_neuron_page = make_html(15)
    # display(HTML(test_neuron_page))
    f = open("/workspace/_scratch/test_neuron_page.html", "w")
    f.write(test_neuron_page)
    f.close()
# %%
if debug:
    num_pages = 5
else:
    num_pages = model.cfg.d_mlp
for neuron_index in tqdm.tqdm(range(num_pages)):
    with open(WEBSITE_DIR / f"{neuron_index}.html", "w") as f:
        f.write(make_html(neuron_index))
# %%
def get_num_layers_generated_model(model_name):
    cfg = loading.get_pretrained_model_config(model_name)
    MODEL_DIR = REAL_DIR/model_name
    num_layers_generated = len(list(MODEL_DIR.iterdir()))
    num_neurons_generated_final = len(list(REAL_DIR/model_name/str(num_layers_generated - 1)).iterdir())
    if num_neurons_generated_final < cfg.d_mlp:
        # Subtract one, final layer incomplete
        return num_layers_generated - 1
    else:
        return num_layers_generated


def gen_index_page(model_names):
    htmls = []
    htmls.append(
        "<h1>Neuroscope: A Website for Mechanistic Interpretability of Language Models</h1>"
    )
    htmls.append(
        "<div>Each model has a page per neuron, displaying the top 20 maximum activating dataset examples.</div>"
    )
    htmls.append("<h2>Supported models</h2>")
    htmls.append("<ul>")
    for name in model_names:
        fancy_name = get_fancy_model_name(name)
        htmls.append(
            f"<li><b><a href='./{name}/index.html'>{name}</a>:</b> {fancy_name}. Dataset: {fancy_data_names[model_name_to_data_name(name)]}</li>"
        )
    htmls.append("</ul>")
    return "\n".join(htmls)


def gen_model_page(model_name):
    cfg = loading.get_pretrained_model_config(model_name)
    layer_num = get_num_layers_generated_model(model_name)
    # if layer_num < 
    # layer_num = int(re.match("solu-(\d+)l.*", model_name).group(1))
    htmls = []
    htmls.append(f"<div style='font-size:medium;'>")
    if neuron_index > 0:
        htmls.append(f"< <a href='./0/0.html'>First Neuron</a> | ")
    htmls.append(f"<a href='../index.html'>Home</a> | ")
    htmls.append(
        f"<a href='./{random.randint(0, layer_num-1)}/{random.randint(0, model.cfg.d_mlp-1)}.html'>Random</a> | "
    )
    if neuron_index < model.cfg.d_mlp:
        htmls.append(
            f"<a href='./{layer_num-1}/{neuron_index+1}.html'>Final Neuron</a> >"
        )
    htmls.append(f"</div>")
    htmls.append(f"<h1>Model Index Page: {get_fancy_model_name(model_name)}</h1>")
    data_name = model_name_to_data_name(model_name)
    
    htmls.append(f"<h1>Dataset: {fancy_data_names[data_name]}</h1>")
    
    htmls.append(
        f"<h2>Hooked Transformer Loading: <span style='font-family: \"Courier New\"'>HookedTransformer.from_pretrained('{model_name}')</span></h2>"
    )
    htmls.append(f"<h2>Layers:</h2>")
    htmls.append("<ul>")
    for l in range(cfg.n_layers):
        if l < layer_num:
            htmls.append(f"<li><b><a href='./{l}/0.html'>Layer #{l}</b></a></li>")
        else:
            htmls.append(f"<li>Layer #{l} (Not yet added)</li>")

    htmls.append("</ul>")
    return "\n".join(htmls)

# %%
MAKE_META_FILES = False

REDIRECT_TO_INDEX = """
<!DOCTYPE html>
<html>
<head>
    <script>
        window.location.replace("index.html");
    </script>
</head>
</html>
"""

if IN_IPYTHON and MAKE_META_FILES:
    REAL_DIR = Path(f"/workspace/neuroscope/v{website_version}")
    REAL_DIR.mkdir(exist_ok=True)
    model_names = list(map(lambda k: k.name, list(REAL_DIR.iterdir())))
    model_names.sort(key = lambda name: loading.get_pretrained_model_config(name).n_params)
    model_names.sort(key = lambda name: model_name_to_data_name(name))
    print(model_names)
# %%
if IN_IPYTHON and MAKE_META_FILES:
    print("Index:")
    index_html = gen_index_page(model_names)
    display(HTML(index_html))
    (REAL_DIR / "index.html").write_text(index_html)
    for name in model_names:
        print(f"Model: {name}")
        model_html = gen_model_page(name)
        display(HTML(model_html))
        folder = REAL_DIR / name
        folder.mkdir(exist_ok=True)
        (folder / "index.html").write_text(model_html)
        (folder / "model.html").write_text(REDIRECT_TO_INDEX)
command = "scp -r /workspace/neuroscope/v4/* neelnanda_lexoscope@ssh.phx.nearlyfreespeech.net:/home/public/"
# scp -r /workspace/neuroscope/v4/gpt2-large neelnanda_lexoscope@ssh.phx.nearlyfreespeech.net:/home/public/gpt2-large
# scp -r /workspace/neuroscope/v4/gpt2-medium neelnanda_lexoscope@ssh.phx.nearlyfreespeech.net:/home/public/gpt2-medium
# scp -r /workspace/neuroscope/v4/gpt2-small neelnanda_lexoscope@ssh.phx.nearlyfreespeech.net:/home/public/gpt2-small
# Note to self: Password kept in Pwd Manager in browser


# %%
text = "Mechanistic Interpretability (MI) is the study of reverse engineering neural networks~ Taking an inscrutable stack of matrices where we know that it works, and trying to reverse engineer how it works~ And often this inscrutable stack of matrices can be decompiled to a human interpretable algorithm~ In my (highly biased) opinion, this is one of the most exciting research areas in ML~"