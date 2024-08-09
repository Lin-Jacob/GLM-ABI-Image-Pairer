import os
from flask import Flask, render_template_string, send_from_directory

app = Flask(__name__)

output_dir = "matched_output"

matched_images = [file for file in os.listdir(output_dir) if file.endswith(".png")]

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Matched ABI and GLM Images</title>
</head>
<body>
    <h1>Matched ABI and GLM Images</h1>
    <ul>
        {% for image_file in matched_images %}
        <li><img src="{{ url_for('serve_image', filename=image_file) }}" alt="{{ image_file }}"></li>
        {% endfor %}
    </ul>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html_template, matched_images=matched_images)

@app.route('/matched_output/<filename>')
def serve_image(filename):
    return send_from_directory(output_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
