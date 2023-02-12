"""A file for template strings to be filled out elsewhere
"""

# Redirects to the index. Used as model_name/model.html for backward compatibility reasons
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

# Redirects to the index one level up. Used as model_name/layer/index.html
REDIRECT_TO_INDEX_ONE_UP = """
<!DOCTYPE html>
<html>
<head>
    <script>
        window.location.replace("../index.html");
    </script>
</head>
</html>
"""

# Redirects to a random neuron page for a single model layer
RANDOM_REDIRECT_1D = """
<!DOCTYPE html>
<html>

<head>
    <script>
        window.location.replace(Math.floor(Math.random() * {neuron}) + ".html");
    </script>
</head>

</html>
    """

# Redirects to a random layer/neuron page
RANDOM_REDIRECT_2D = """
<!DOCTYPE html>
<html>

<head>
    <script>
        window.location.replace(Math.floor(Math.random() * {layer}) + "/" + Math.floor(Math.random() * {neuron}) + ".html");
    </script>
</head>

</html>
    """

NEUROSCOPE_MAIN_INDEX_TABLE_HEADINGS = """<tr style="text-align: right;">
      <th>Model</th>
      <th>Random</th>
      <th>Act Fn</th>
      <th>Dataset</th>
      <th>Layers</th>
      <th>Neurons per Layer</th>
      <th>Total Neurons</th>
      <th>Params</th>
    </tr>"""

NEUROSCOPE_MAIN_INDEX = """
<h1>Neuroscope: A Website for Mechanistic Interpretability of Language Models</h1>
<div>Each model has a page per neuron, displaying the top 20 maximum activating dataset examples.</div>
<div>See <a href="https://neelnanda.io/neuroscope-docs">the documentation</a> for more.</div>
<h2>Supported models</h2>
{models_table}
"""

MODEL_INDEX = """
<div style='font-size:medium;'>
< <a href='./0/0.html'>First Neuron</a> | 
<a href='../index.html'>Home</a> | 
<a href='random.html'>Random</a> | 
<a href='./{max_layer}/{max_neuron}.html'>Final Neuron</a> >
</div>
<h1>Model Index Page: {fancy_model_name}</h1>
<h1>Dataset: {fancy_data_name}</h1>
<h2>Hooked Transformer Loading: <span style='font-family: "Courier New"'>{model_name}</span></h2>
<h2>Layers:</h2>
{model_layers_table}
"""