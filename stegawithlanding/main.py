from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import os
import numpy as np
from lsb_aes_utils import *

app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/tools')
def tools():
    return render_template('index.html')

@app.route('/hide', methods=['POST'])
def hide():
    image_file = request.files.get('image')
    message = request.form.get('secret')
    key = request.form.get('key')
    algorithm = request.form.get('algorithm', 'lsb')

    if not image_file or not message:
        flash("Image and message are required.")
        return redirect(url_for('tools'))

    try:
        image = Image.open(image_file)
        stego_image = hide_message_in_image(image, message, key, algorithm)
    except Exception as e:
        return f"Error hiding message: {str(e)}"

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'stego_{algorithm}.png')
    stego_image.save(output_path)
    return send_file(output_path, as_attachment=True)

@app.route('/extract', methods=['POST'])
def extract():
    image_file = request.files.get('image')
    key = request.form.get('key')
    algorithm = request.form.get('algorithm', 'lsb')

    if not image_file or image_file.filename == '':
        return "No image uploaded."

    filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(image_path)

    try:
        message = extract_message_lsb(image_path, key, algorithm)
    except Exception as e:
        return f"Error extracting message: {str(e)}"

    return f"<h2>Extracted Message ({algorithm.upper()}:</h2><p>{message}</p>"

@app.route('/capacity', methods=['POST'])
def calculate_image_capacity():
    image_file = request.files.get('image')
    algorithm = request.form.get('algorithm', 'lsb')

    if not image_file:
        return "No image uploaded."

    try:
        image = Image.open(image_file)
        capacity = calculate_capacity(image, algorithm)
        return f"<h3>Capacity for {algorithm.upper()}:</h3><p>{capacity} bytes ({capacity} characters)</p>"
    except Exception as e:
        return f"Error calculating capacity: {str(e)}"

@app.route('/hide_image', methods=['POST'])
def hide_image():
    cover_file = request.files.get('cover_image')
    secret_file = request.files.get('secret_image')
    key = request.form.get('key')
    algorithm = request.form.get('algorithm', 'lsb_image')

    if not cover_file or not secret_file:
        flash("Both cover and secret images are required.")
        return redirect(url_for('tools'))

    try:
        cover_image = Image.open(cover_file)
        secret_image = Image.open(secret_file)
        stego_image = hide_image_in_image(cover_image, secret_image, key, algorithm)
    except Exception as e:
        return f"Error hiding image: {str(e)}"

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'stego_image_{algorithm}.png')
    stego_image.save(output_path)
    return send_file(output_path, as_attachment=True, download_name=f'hidden_image_{algorithm}.png')

@app.route('/extract_image', methods=['POST'])
def extract_image():
    image_file = request.files.get('image')
    key = request.form.get('key')
    algorithm = request.form.get('algorithm', 'lsb_image')

    if not image_file or image_file.filename == '':
        return "No image uploaded."

    filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(image_path)

    try:
        extracted_image = extract_image_from_image(image_path, key, algorithm)
    except Exception as e:
        return f"Error extracting image: {str(e)}"

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'extracted_image_{algorithm}.png')
    extracted_image.save(output_path)
    return send_file(output_path, as_attachment=True, download_name=f'extracted_image_{algorithm}.png')

@app.route('/visualize_message', methods=['POST'])
def visualize_message():
    image_file = request.files.get('image')
    message = request.form.get('secret')
    key = request.form.get('key')
    algorithm = request.form.get('algorithm', 'lsb')

    if not image_file or not message:
        return "Image and message are required."

    try:
        image = Image.open(image_file)
        stego_image, visualization = visualize_message_hiding(image, message, key, algorithm)

        # Save both images
        stego_path = os.path.join(app.config['UPLOAD_FOLDER'], f'stego_{algorithm}.png')
        vis_path = os.path.join(app.config['UPLOAD_FOLDER'], f'visualization_{algorithm}.png')

        stego_image.save(stego_path)
        visualization.save(vis_path)

        return render_template('visualization.html', 
                             stego_image=f'stego_{algorithm}.png',
                             visualization=f'visualization_{algorithm}.png',
                             algorithm=algorithm.upper(),
                             type='Message')
    except Exception as e:
        return f"Error creating visualization: {str(e)}"

@app.route('/visualize_image', methods=['POST'])
def visualize_image():
    cover_file = request.files.get('cover_image')
    secret_file = request.files.get('secret_image')
    key = request.form.get('key')
    algorithm = request.form.get('algorithm', 'lsb_image')

    if not cover_file or not secret_file:
        return "Both cover and secret images are required."

    try:
        cover_image = Image.open(cover_file)
        secret_image = Image.open(secret_file)
        stego_image, visualization = visualize_image_hiding(cover_image, secret_image, key, algorithm)

        # Save both images
        stego_path = os.path.join(app.config['UPLOAD_FOLDER'], f'stego_image_{algorithm}.png')
        vis_path = os.path.join(app.config['UPLOAD_FOLDER'], f'visualization_image_{algorithm}.png')

        stego_image.save(stego_path)
        visualization.save(vis_path)

        return render_template('visualization.html', 
                             stego_image=f'stego_image_{algorithm}.png',
                             visualization=f'visualization_image_{algorithm}.png',
                             algorithm=algorithm.upper(),
                             type='Image')
    except Exception as e:
        return f"Error creating visualization: {str(e)}"

@app.route('/xor_compare', methods=['POST'])
def xor_compare():
    original_file = request.files.get('original_image')
    stego_file = request.files.get('stego_image')
    technique = request.form.get('technique', 'unknown')

    if not original_file or not stego_file or original_file.filename == '' or stego_file.filename == '':
        flash("Both original and steganographic images are required.")
        return redirect(url_for('tools'))

    try:
        # Ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Load and validate images with size limits for deployment
        original_image = Image.open(original_file)
        stego_image = Image.open(stego_file)

        # Resize large images to prevent memory issues in deployment
        max_size = (1024, 1024)
        if original_image.size[0] > max_size[0] or original_image.size[1] > max_size[1]:
            original_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        if stego_image.size[0] > max_size[0] or stego_image.size[1] > max_size[1]:
            stego_image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Ensure both images are same size
        if original_image.size != stego_image.size:
            stego_image = stego_image.resize(original_image.size, Image.Resampling.LANCZOS)

        # Convert to RGB to ensure consistency
        original_image = original_image.convert('RGB')
        stego_image = stego_image.convert('RGB')

        # Analyze the type of changes to better identify the technique
        original_pixels = np.array(original_image)
        stego_pixels = np.array(stego_image)

        # Count different types of changes
        total_changes = np.sum(original_pixels != stego_pixels)
        lsb_changes = 0

        for channel in range(3):
            orig_lsb = original_pixels[:,:,channel] & 1
            stego_lsb = stego_pixels[:,:,channel] & 1
            lsb_changes += np.sum(orig_lsb != stego_lsb)

        # Auto-detect technique if not specified or unknown
        if technique == 'unknown' or not technique:
            if lsb_changes > 0 and lsb_changes == total_changes:
                # Only LSB changes detected
                if lsb_changes < original_pixels.size * 0.1:
                    technique = 'spread_spectrum'  # Sparse LSB changes
                else:
                    technique = 'lsb'  # Dense LSB changes
            elif total_changes > lsb_changes:
                technique = 'dct'  # Non-LSB changes detected
            else:
                technique = 'lsb'  # Default

        # Create enhanced XOR visualization
        visualization = create_bit_visualization(original_image, stego_image, technique)

        # Calculate statistics for display
        change_percentage = (total_changes / original_pixels.size) * 100 if original_pixels.size > 0 else 0

        # Save images with error handling
        try:
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_compare.png')
            stego_path = os.path.join(app.config['UPLOAD_FOLDER'], 'stego_compare.png')
            vis_path = os.path.join(app.config['UPLOAD_FOLDER'], 'xor_comparison.png')

            original_image.save(original_path, 'PNG', optimize=True)
            stego_image.save(stego_path, 'PNG', optimize=True)
            visualization.save(vis_path, 'PNG', optimize=True)
        except Exception as save_error:
            app.logger.error(f"Error saving XOR comparison images: {str(save_error)}")
            flash(f"Error saving comparison images: {str(save_error)}")
            return redirect(url_for('tools'))

        app.logger.info(f"XOR Analysis Complete - Technique: {technique}, Changes: {total_changes}, LSB Changes: {lsb_changes}")

        return render_template('xor_comparison.html',
                             original_image='original_compare.png',
                             stego_image='stego_compare.png',
                             xor_visualization='xor_comparison.png',
                             technique=technique.upper(),
                             total_changes=total_changes,
                             lsb_changes=lsb_changes,
                             change_percentage=f"{change_percentage:.4f}")
    except MemoryError:
        app.logger.error("Memory error during XOR comparison - images too large")
        flash("Images are too large for comparison. Please use smaller images.")
        return redirect(url_for('tools'))
    except Exception as e:
        app.logger.error(f"XOR Comparison Error: {str(e)}")
        flash(f"Error creating XOR comparison: {str(e)}")
        return redirect(url_for('tools'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)