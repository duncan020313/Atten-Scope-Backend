from datetime import datetime
from flask import (
    Flask,
    request,
    render_template,
    render_template_string,
    url_for,
)
from markupsafe import escape
import os
from inference import get_hooked_model, get_htmls

app = Flask(__name__)

# Ensure the 'generated_html' directory exists
if not os.path.exists("generated_html"):
    os.makedirs("generated_html")

# Load models
models = {
    "gpt2": get_hooked_model("gpt2"),
    "gpt2-large": get_hooked_model("gpt2-large"),
    "codellama-7b-python": get_hooked_model("codellama/CodeLlama-7b-Python-hf"),
}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/generate_html", methods=["POST"])
def generate_html():
    prompt = request.form["prompt"]
    model_name = request.form["model"]
    sanitized_prompt = prompt
    sanitized_model_name = escape(model_name)

    # Generate text using the selected model
    hooked_model = models.get(model_name)
    htmls = get_htmls(sanitized_prompt, hooked_model)

    # Simulate multiple HTML generation
    gen_path = "generated_html"
    save_path = f"old_versions/{datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(save_path, exist_ok=True)
    generated_files = []
    for i, generated_text in enumerate(htmls):
        file_name = f"Layer_{i}.html"
        # Create dir for save_path
        html_content = generated_text

        gen_path_full = os.path.join(gen_path, file_name)
        save_path_full = os.path.join(save_path, file_name)

        with open(gen_path_full, "w") as f:
            f.write(html_content)
        with open(save_path_full, "w") as f:
            f.write(html_content)

        generated_files.append(file_name)

    # Generate buttons for each HTML file
    buttons_html = "".join(
        [
            f'<div class="button-container"><a href="{url_for("serve_html", filename=filename)}"><button>{filename}</button></a></div>'
            for filename in generated_files
        ]
    )

    return render_template_string(
        f"<html><head><link rel='stylesheet' href='/static/styles/home.css'></head><body class='styled-body'>{buttons_html}</body></html>"
    )


@app.route("/generated_html/<filename>")
def serve_html(filename):
    file_path = os.path.join("generated_html", filename)
    with open(file_path, "r") as f:
        return f.read()


@app.route("/old_versions", methods=["GET"])
def old_versions():
    dir_list = os.listdir("old_versions")

    # Sort with date
    def find_date(file_name):
        return datetime.strptime(file_name, f"%Y-%m-%d_%H-%M-%S")

    sorted_dir_list = sorted(dir_list, key=find_date)

    atag_htmls = "".join(
        [
            f'<div class="button-container"><a href="{url_for("serve_old_version", filename=filename)}"><button>{filename}</button></a></div>'
            for filename in sorted_dir_list
        ]
    )
    return render_template_string(
        f"<html><head><link rel='stylesheet' href='/static/styles/home.css'></head><body class='styled-body'><h1>Old Versions</h1>{atag_htmls}</body></html>"
    )


@app.route("/old_versions/<filename>")
def serve_old_version(filename):
    file_path = os.path.join("old_versions", filename)
    file_list = os.listdir(file_path)

    def find_layer(file_name):
        return int(file_name.split("_")[1].split(".")[0])

    file_list = sorted(file_list, key=find_layer)
    atag_htmls = "".join(
        [
            f'<div class="button-container"><a href="{filename}/{file}"><button>{file}</button></a></div>'
            for file in file_list
        ]
    )
    return render_template_string(
        f"<html><head><link rel='stylesheet' href='/static/styles/home.css'></head><body class='styled-body'><h1>{filename}</h1>{atag_htmls}</body></html>"
    )


@app.route("/old_versions/<filename>/<file>")
def serve_old_version_file(filename, file):
    file_path = os.path.join("old_versions", filename, file)
    with open(file_path, "r") as f:
        return f.read()


if __name__ == "__main__":
    app.run(debug=True)
