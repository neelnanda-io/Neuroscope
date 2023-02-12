from neel.imports import *

import neuroscope.templates as templates
import neuroscope.utils as nutils
import neuroscope


website_version = 4
MAKE_META_FILES = True
WEBSITE_DIR = Path(f"/workspace/neuroscope/v{website_version}")
WEBSITE_DIR.mkdir(exist_ok=True)
# %%
try:
    close_scp()
except:
    pass
import paramiko

# Create an SSH client
client = paramiko.SSHClient()

# Automatically add the server's host key (useful for the first time you connect to the server)
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Connect to the remote server
client.connect(hostname='ssh.phx.nearlyfreespeech.net', username='neelnanda_lexoscope', password='eeckui7UEWaa')

# Use the SCP client to copy the file to the remote server
scp = client.open_sftp()
# scp.put('/workspace/solu_project/solu/microscope/scratch_remote_code.py', '/home/public/code/scratch_remote_code.py')

def close_scp():
    scp.close()
    client.close()

# %%
def get_num_sub_folders(path):
    path = Path(path)
    return len(list(filter(lambda name: name.is_dir(), path.iterdir())))

def get_list_sub_folders(path):
    path = Path(path)
    return (list(filter(lambda name: name.is_dir(), path.iterdir())))

@lru_cache(maxsize=None)
def get_config(model_name):
    return loading.get_pretrained_model_config(model_name)


def make_random_redirect(neuron):
    return templates.RANDOM_REDIRECT_1D.format(neuron=neuron)

def make_random_redirect_2d(layer, neuron):
    return f"""
<!DOCTYPE html>
<html>

<head>
    <script>
        window.location.replace(Math.floor(Math.random() * {layer}) + "/" + Math.floor(Math.random() * {neuron}) + ".html");
    </script>
</head>

</html>
    """
# print(get_list_sub_folders(WEBSITE_DIR/"solu-12l"))
# %%
def model_name_to_data_name(model_name):
    if "old" in model_name or "pile" in model_name:
        data_name = "pile"
    elif "gpt" in model_name:
        data_name = "openwebtext"
    elif model_name.startswith("solu") or model_name.startswith("gelu"):
        data_name = "c4-code"
    return data_name

fancy_data_names = {
    "c4-code": "80% C4 (Web Text) and 20% Python Code",
    "c4": "C4 (Web Text)",
    "code": "Python Code",
    "pile": "The Pile",
    "openwebtext": "Open Web Text",
}

def get_fancy_model_name(model_name):
    cfg = get_config(model_name)
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
def get_num_layers_generated_model(model_name):
    cfg = get_config(model_name)
    MODEL_DIR = WEBSITE_DIR/model_name
    num_layers_generated = get_num_sub_folders(MODEL_DIR)
    num_neurons_generated_final = len(list((WEBSITE_DIR/model_name/str(num_layers_generated - 1)).iterdir()))
    if num_neurons_generated_final < cfg.d_mlp:
        # Subtract one, final layer incomplete
        return num_layers_generated - 2
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
    # htmls.append("<ul>")
    columns = ["Model", "Random", "Act Fn", "Dataset", "Layers", "Neurons per Layer", "Total Neurons", "Params"]
    rows = []
    for name in model_names:
        cfg = get_config(name)
        num_layers = get_num_layers_generated_model(name)
        rows.append([
            f"<a href='./{name}/index.html'>{name}</a>",
            f"<a href='./{name}/random.html'>Random</a>",
            cfg.act_fn[:4],
            fancy_data_names[model_name_to_data_name(name)],
            cfg.n_layers,
            cfg.d_mlp,
            cfg.d_mlp * cfg.n_layers,
            cfg.n_params,
        ])
    df = pd.DataFrame(rows, columns=columns)
    #     fancy_name = get_fancy_model_name(name)
    #     htmls.append(
    #         f"<li><b><a href='./{name}/index.html'>{name}</a>:</b> {fancy_name}. Dataset: {fancy_data_names[model_name_to_data_name(name)]}</li>"
    #     )
    # htmls.append("</ul>")
    htmls.append(df.to_html(index=False, escape=False))
    return "\n".join(htmls)


def gen_model_page(model_name):
    cfg = get_config(model_name)
    layer_num = get_num_layers_generated_model(model_name)
    # if layer_num < 
    # layer_num = int(re.match("solu-(\d+)l.*", model_name).group(1))
    htmls = []
    htmls.append(f"<div style='font-size:medium;'>")
    htmls.append(f"< <a href='./0/0.html'>First Neuron</a> | ")
    htmls.append(f"<a href='../index.html'>Home</a> | ")
    htmls.append(
        f"<a href='random.html'>Random</a> | "
    )
    htmls.append(
        f"<a href='./{layer_num-1}/{cfg.d_mlp - 1}.html'>Final Neuron</a> >"
    )
    htmls.append(f"</div>")
    htmls.append(f"<h1>Model Index Page: {get_fancy_model_name(model_name)}</h1>")
    data_name = model_name_to_data_name(model_name)
    
    htmls.append(f"<h1>Dataset: {fancy_data_names[data_name]}</h1>")
    
    htmls.append(
        f"<h2>Hooked Transformer Loading: <span style='font-family: \"Courier New\"'>HookedTransformer.from_pretrained('{model_name}')</span></h2>"
    )
    htmls.append(f"<h2>Layers:</h2>")
    if layer_num < cfg.n_layers:
        htmls.append(f"<h3>Only generated pages up to layer {layer_num - 1}</h3>")

    rows = []
    for l in range(layer_num):
        rows.append(
            [
                f"<b>Layer #{l}</b>",
                f"<a href='./{l}/0.html'>First</a>",
                f"<a href='./{l}/random.html'>Random</a>",
                f"<a href='./{l}/{cfg.d_mlp - 1}.html'>Last</a>",
            ]
        )
    df = pd.DataFrame(rows, columns=["Layer", "First", "Random", "Last"])
    htmls.append(df.to_html(escape=False, header=False, index=False))
    # htmls.append("<ul>")
    # for l in range(cfg.n_layers):
    #     if l < layer_num:
    #         htmls.append(f"<li><b><a href='./{l}/0.html'>Layer #{l}</b></a></li>")
    #     else:
    #         htmls.append(f"<li>Layer #{l} (Not yet added)</li>")

    # htmls.append("</ul>")
    return "\n".join(htmls)

display(HTML(gen_model_page("solu-12l")))
display(HTML(gen_index_page(["gpt2-xl", "solu-12l", "solu-1l", "solu-2l"])))

# %%
def make_file(path, text):
    path.write_text(text)
    remote_path = Path("/home/public")/path.relative_to(WEBSITE_DIR)
    try:
        scp.put(str(path), str(remote_path))
    except:
        print(f"Failed to upload {path} to {remote_path}")

def make_all_files(model_names):
    dir_names = scp.listdir()
    for name in model_names:
        if name not in dir_names:
            scp.mkdir(name)
            print(name)


# %%

if IN_IPYTHON and MAKE_META_FILES:
    model_names = list(map(lambda k: k.name, get_list_sub_folders(WEBSITE_DIR)))
    model_names.sort(key = lambda name: get_config(name).n_params)
    model_names.sort(key = lambda name: model_name_to_data_name(name))
    print(model_names)
    d = {}
    for model_name in model_names:
        cfg = get_config(model_name)
        d[model_name] = [cfg.n_layers, get_num_layers_generated_model(model_name), cfg.d_mlp]
    print(d)
    display(HTML(gen_index_page(model_names)))
# %%
if IN_IPYTHON and MAKE_META_FILES:
    print("Index:")
    index_html = gen_index_page(model_names)
    display(HTML(index_html))
    make_file((WEBSITE_DIR / "index.html"), (index_html))
    # (WEBSITE_DIR / "index.html").write_text(index_html)
    for name in model_names:
        print("Starting model", name)
        cfg = get_config(name)
        num_layers = get_num_layers_generated_model(name)
        print("Num Layers:", num_layers)
        print("Total Layers:", cfg.n_layers)
        print("Num Neurons:", cfg.d_mlp)
        print(f"Model: {name}")
        model_html = gen_model_page(name)
        display(HTML(model_html))
        folder = WEBSITE_DIR / name
        assert folder.exists()
        # folder.mkdir(exist_ok=True)
        make_file((folder / "index.html"), (model_html))
        # (folder / "index.html").write_text(model_html)
        make_file((folder / "model.html"), (REDIRECT_TO_INDEX))
        # (folder / "model.html").write_text(REDIRECT_TO_INDEX)
        make_file((folder / "random.html"), (make_random_redirect_2d(num_layers, cfg.d_mlp)))
        # (folder / "random.html").write_text(make_random_redirect_2d(num_layers, cfg.d_mlp))
        for subfolder in get_list_sub_folders(folder):
            make_file((subfolder / "random.html"), (make_random_redirect(cfg.d_mlp)))
            # (subfolder / "random.html").write_text(make_random_redirect(cfg.d_mlp))
            make_file((subfolder/"index.html"), (REDIRECT_TO_INDEX_ONE_UP))
            # (subfolder/"index.html").write_text(REDIRECT_TO_INDEX_ONE_UP)
            
# %%
