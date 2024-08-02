import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, send_from_directory

app = Flask(__name__)

directory = "data"
output_dir = "images"  
os.makedirs(output_dir, exist_ok=True)

def visualize_nc_file(file_path, output_dir, variable_name):
    # Read the .nc file
    dataset = netCDF4.Dataset(file_path)

    if variable_name not in dataset.variables:
        print(f"Variable '{variable_name}' not found in {file_path}")
        return None

    data = dataset.variables[variable_name][:]

    # Flip GLM data on Y-axis
    if "GLM" in file_path:
        data = np.flip(data, axis=0)

    plt.figure(figsize=(data.shape[1]/100, data.shape[0]/100)) 
    plt.imshow(data, cmap='gray')
    plt.axis('off') 
    plt.tight_layout()

    # Save the image
    image_file = os.path.join(output_dir, os.path.basename(file_path) + '.png')
    plt.savefig(image_file, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

    return image_file

nc_files = [file for file in os.listdir(directory) if file.endswith(".nc")]

image_files = []
for nc_file in nc_files:
    file_path = os.path.join(directory, nc_file)
    if "ABI" in nc_file:
        image_file = visualize_nc_file(file_path, output_dir, "radiance")
    elif "GLM" in nc_file:
        image_file = visualize_nc_file(file_path, output_dir, "GLM_data")
    if image_file:
        image_files.append(image_file)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>ABI and GLM Images</title>
</head>
<body>
    <h1>Generated Images from .nc Files</h1>
    <h2>ABI Images</h2>
    <ul>
        {% for image_file in abi_images %}
        <li><img src="{{ url_for('serve_image', filename=image_file) }}" alt="{{ image_file }}"></li>
        {% endfor %}
    </ul>

    <h2>GLM Images</h2>
    <ul>
        {% for image_file in glm_images %}
        <li><img src="{{ url_for('serve_image', filename=image_file) }}" alt="{{ image_file }}"></li>
        {% endfor %}
    </ul>
</body>
</html>
"""

@app.route('/')
def index():
    abi_images = [os.path.basename(img) for img in image_files if "ABI" in img]
    glm_images = [os.path.basename(img) for img in image_files if "GLM" in img]
    return render_template_string(html_template, abi_images=abi_images, glm_images=glm_images)

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(output_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
