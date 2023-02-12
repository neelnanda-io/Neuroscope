"""A file for template strings to be filled out elsewhere
"""

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