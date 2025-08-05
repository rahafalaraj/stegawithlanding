
import os
import numpy as np
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
import struct
from math import log2
from scipy import fftpack
import random

def encrypt_message(message, key):
    cipher = AES.new(key, AES.MODE_CBC)
    encrypted_data = cipher.iv + cipher.encrypt(pad(message.encode(), AES.block_size))
    return encrypted_data

def decrypt_message(encrypted_data, key):
    cipher = AES.new(key, AES.MODE_CBC, iv=encrypted_data[:16])
    decrypted_data = unpad(cipher.decrypt(encrypted_data[16:]), AES.block_size)
    return decrypted_data.decode()

def calculate_entropy(image):
    pixels = np.array(image)
    total_pixels = pixels.size
    unique, counts = np.unique(pixels, return_counts=True)
    probabilities = counts / total_pixels
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    return entropy

def calculate_local_randomness(image):
    pixels = np.array(image)
    changes = np.count_nonzero(np.diff(pixels, axis=0)) + np.count_nonzero(np.diff(pixels, axis=1))
    total_pairs = (pixels.shape[0] - 1) * pixels.shape[1] + (pixels.shape[1] - 1) * pixels.shape[0]
    return changes / total_pairs if total_pairs else 0

def is_solid_color(image):
    pixels = np.array(image)
    return np.all(pixels == pixels[0, 0])

def is_gradient(image, threshold=1.0):
    pixels = np.array(image).astype(np.int32)
    row_diffs = np.abs(np.diff(pixels, axis=0)).mean()
    col_diffs = np.abs(np.diff(pixels, axis=1)).mean()
    return row_diffs < threshold and col_diffs < threshold

def is_image_suitable(image):
    bw_image = image.convert('L')

    if is_solid_color(bw_image):
        return False
    if is_gradient(bw_image):
        return False

    entropy = calculate_entropy(bw_image)
    local_randomness = calculate_local_randomness(bw_image)
    return entropy > 0.5 and local_randomness > 0.2

def calculate_capacity(image, algorithm='lsb'):
    """Calculate maximum message capacity for different algorithms"""
    width, height = image.size
    total_pixels = width * height * 3  # RGB channels

    if algorithm == 'lsb':
        # LSB can store 1 bit per color channel
        capacity_bits = total_pixels
    elif algorithm == 'dct':
        # DCT uses 8x8 blocks, can store ~1 bit per block
        blocks_x = width // 8
        blocks_y = height // 8
        capacity_bits = blocks_x * blocks_y * 3  # RGB channels

    elif algorithm == 'spread_spectrum':
        # Spread spectrum uses multiple pixels per bit
        capacity_bits = total_pixels // 8
    else:
        capacity_bits = total_pixels

    # Account for length header (32 bits)
    capacity_bits -= 32
    capacity_bytes = capacity_bits // 8

    return max(0, capacity_bytes)

def create_bit_visualization(original_image, stego_image, algorithm='lsb'):
    """Create a high-contrast visualization showing where bits were modified with amplified visibility"""
    # Ensure both images are the same size
    original_image = original_image.convert('RGB')
    stego_image = stego_image.convert('RGB')

    # Resize stego image to match original if needed
    if original_image.size != stego_image.size:
        stego_image = stego_image.resize(original_image.size, Image.Resampling.LANCZOS)

    original_pixels = np.array(original_image)
    stego_pixels = np.array(stego_image)

    # Start with pure black background
    vis_image = np.zeros_like(original_pixels)

    # Create different visualizations based on algorithm
    if algorithm in ['spread_spectrum', 'dct']:
        # For sparse algorithms, show any pixel differences
        difference_mask = np.any(original_pixels != stego_pixels, axis=2)

        if algorithm == 'spread_spectrum':
            # Bright neon red for spread spectrum with enhanced visibility
            vis_image[difference_mask] = [255, 0, 80]
        elif algorithm == 'dct':
            # Bright neon green for DCT with enhanced visibility
            vis_image[difference_mask] = [0, 255, 80]
        else:
            # Pure white for unknown
            vis_image[difference_mask] = [255, 255, 255]

    else:
        # For LSB-based algorithms, create a more comprehensive analysis
        any_changes = np.zeros_like(original_pixels[:,:,0], dtype=bool)

        # Check for ANY bit differences, not just LSB
        for channel in range(3):
            channel_diff = original_pixels[:,:,channel] != stego_pixels[:,:,channel]
            any_changes |= channel_diff

        # Also specifically check LSB changes for different coloring
        lsb_only_changes = np.zeros_like(original_pixels[:,:,0], dtype=bool)
        for channel in range(3):
            orig_lsb = original_pixels[:,:,channel] & 1
            stego_lsb = stego_pixels[:,:,channel] & 1
            lsb_only_changes |= (orig_lsb != stego_lsb)

        # Color different types of changes differently
        if algorithm == 'lsb':
            # Pure bright cyan for LSB changes
            vis_image[lsb_only_changes] = [0, 255, 255]
            # Bright yellow for any non-LSB changes
            non_lsb_changes = any_changes & ~lsb_only_changes
            vis_image[non_lsb_changes] = [255, 255, 0]
        elif algorithm == 'lsb_image':
            # Bright magenta for image hiding
            vis_image[lsb_only_changes] = [255, 0, 255]
            # Orange for non-LSB changes
            non_lsb_changes = any_changes & ~lsb_only_changes
            vis_image[non_lsb_changes] = [255, 165, 0]
        elif algorithm == 'random_lsb':
            # Bright lime green for random LSB
            vis_image[lsb_only_changes] = [128, 255, 0]
            # Red for non-LSB changes
            non_lsb_changes = any_changes & ~lsb_only_changes
            vis_image[non_lsb_changes] = [255, 0, 0]
        else:
            # Default - pure white for all changes
            vis_image[any_changes] = [255, 255, 255]

    # Amplify the visualization by creating a larger version (3x scale for better visibility)
    height, width = vis_image.shape[:2]

    # Create an amplified version (3x scale with enhanced contrast)
    amplified_vis = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)

    # Fill the amplified image with enhanced brightness
    for i in range(height):
        for j in range(width):
            if np.any(vis_image[i, j] > 0):  # If there's a change
                # Create a 3x3 block for each changed pixel with maximum brightness
                color = vis_image[i, j].copy()
                # Ensure maximum brightness for visibility
                color = np.clip(color * 1.2, 0, 255).astype(np.uint8)
                amplified_vis[i*3:i*3+3, j*3:j*3+3] = color

    # Add enhanced grid lines for better pixel separation
    # Vertical lines (every 3 pixels)
    for j in range(0, width * 3, 3):
        if j < width * 3:
            amplified_vis[:, j] = np.maximum(amplified_vis[:, j], [96, 96, 96])
            if j + 1 < width * 3:
                amplified_vis[:, j + 1] = np.maximum(amplified_vis[:, j + 1], [48, 48, 48])

    # Horizontal lines (every 3 pixels)
    for i in range(0, height * 3, 3):
        if i < height * 3:
            amplified_vis[i, :] = np.maximum(amplified_vis[i, :], [96, 96, 96])
            if i + 1 < height * 3:
                amplified_vis[i + 1, :] = np.maximum(amplified_vis[i + 1, :], [48, 48, 48])

    return Image.fromarray(amplified_vis)

def visualize_message_hiding(image, message, key=None, algorithm='lsb'):
    """Create visualization for message hiding"""
    original_image = image.copy()
    stego_image = hide_message_in_image(image, message, key, algorithm)
    visualization = create_bit_visualization(original_image, stego_image, algorithm)
    return stego_image, visualization

def visualize_image_hiding(cover_image, secret_image, key=None, algorithm='lsb_image'):
    """Create visualization for image hiding"""
    original_cover = cover_image.copy()
    stego_image = hide_image_in_image(cover_image, secret_image, key, algorithm)
    visualization = create_bit_visualization(original_cover, stego_image, algorithm)
    return stego_image, visualization



def dct_hide_message(image, message, key=None):
    """Hide message using DCT (Discrete Cosine Transform)"""
    image = image.convert('RGB')
    pixels = np.array(image, dtype=float)

    if key:
        key = hashlib.sha256(key.encode()).digest()
        message = encrypt_message(message, key)
    else:
        message = message.encode()

    packed_length = struct.pack('>I', len(message))
    full_message = packed_length + message
    binary_message = ''.join(format(byte, '08b') for byte in full_message)

    bit_index = 0
    height, width = pixels.shape[:2]

    # Process 8x8 blocks
    for channel in range(3):
        for i in range(0, height - 7, 8):
            for j in range(0, width - 7, 8):
                if bit_index >= len(binary_message):
                    break

                # Extract 8x8 block
                block = pixels[i:i+8, j:j+8, channel]

                # Apply DCT
                dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')

                # Modify middle frequency coefficient
                if int(binary_message[bit_index]) == 1:
                    dct_block[4, 4] = abs(dct_block[4, 4]) + 10
                else:
                    dct_block[4, 4] = -(abs(dct_block[4, 4]) + 10)

                # Apply inverse DCT
                idct_block = fftpack.idct(fftpack.idct(dct_block.T, norm='ortho').T, norm='ortho')
                pixels[i:i+8, j:j+8, channel] = np.clip(idct_block, 0, 255)

                bit_index += 1

    return Image.fromarray(pixels.astype(np.uint8))



def spread_spectrum_hide_message(image, message, key=None, spread_factor=16):
    """Hide message using spread spectrum technique"""
    image = image.convert('RGB')
    pixels = np.array(image)

    if key:
        key = hashlib.sha256(key.encode()).digest()
        message = encrypt_message(message, key)
        # Use key for pseudorandom sequence
        master_seed = int.from_bytes(key[:4], 'big')
    else:
        message = message.encode()
        master_seed = 424242  # Default seed

    packed_length = struct.pack('>I', len(message))
    full_message = packed_length + message
    binary_message = ''.join(format(byte, '08b') for byte in full_message)

    flat_pixels = pixels.flatten().astype(int)
    total_pixels = len(flat_pixels)

    if len(binary_message) * spread_factor > total_pixels:
        raise ValueError(f"Message too large ({len(binary_message)} bits) for spread spectrum embedding (needs {len(binary_message) * spread_factor} pixels, have {total_pixels}).")

    # Create a separate random generator for each bit to ensure consistency
    used_positions = set()

    for bit_index, bit in enumerate(binary_message):
        # Use bit-specific seed for consistent position generation
        bit_seed = master_seed + bit_index * 1000
        random.seed(bit_seed)

        # Generate unique positions for this bit
        positions = []
        attempts = 0
        while len(positions) < spread_factor and attempts < spread_factor * 3:
            pos = random.randint(0, total_pixels - 1)
            if pos not in used_positions:
                positions.append(pos)
                used_positions.add(pos)
            attempts += 1

        if len(positions) < spread_factor:
            # Fall back to allowing some overlap if needed
            random.seed(bit_seed)
            positions = [random.randint(0, total_pixels - 1) for _ in range(spread_factor)]

        # Embed bit using majority voting principle
        for pos in positions:
            if int(bit) == 1:
                # Set LSB to 1 with some redundancy
                flat_pixels[pos] = (flat_pixels[pos] & 0xFE) | 1
            else:
                # Set LSB to 0
                flat_pixels[pos] = flat_pixels[pos] & 0xFE

    new_pixels = flat_pixels.reshape(pixels.shape).astype(np.uint8)
    return Image.fromarray(new_pixels)

def hide_message_in_image(image, message, key=None, algorithm='lsb'):
    """Hide message using specified algorithm"""
    if algorithm == 'dct':
        return dct_hide_message(image, message, key)

    elif algorithm == 'spread_spectrum':
        return spread_spectrum_hide_message(image, message, key)
    else:  # Default LSB
        return lsb_hide_message(image, message, key)

def hide_image_in_image(cover_image, secret_image, key=None, algorithm='lsb_image'):
    """Hide image inside another image using specified algorithm"""
    if algorithm == 'random_lsb':
        return random_lsb_hide_image(cover_image, secret_image, key)
    else:  # Default LSB for images
        return lsb_hide_image(cover_image, secret_image, key)

def lsb_hide_image(cover_image, secret_image, key=None):
    """Hide image using LSB technique"""
    cover_image = cover_image.convert('RGB')
    secret_image = secret_image.convert('RGB')

    # Resize secret image to fit in cover image if needed
    cover_width, cover_height = cover_image.size
    secret_width, secret_height = secret_image.size

    # Calculate maximum secret image size (use 1/4 of cover image dimensions)
    max_secret_width = cover_width // 2
    max_secret_height = cover_height // 2

    if secret_width > max_secret_width or secret_height > max_secret_height:
        # Resize while maintaining aspect ratio
        secret_image.thumbnail((max_secret_width, max_secret_height), Image.Resampling.LANCZOS)
        secret_width, secret_height = secret_image.size

    cover_pixels = np.array(cover_image)
    secret_pixels = np.array(secret_image)

    # Store secret image dimensions in first few pixels
    width_bytes = secret_width.to_bytes(4, 'big')
    height_bytes = secret_height.to_bytes(4, 'big')

    # Convert dimensions to binary
    dim_binary = ''.join(format(byte, '08b') for byte in width_bytes + height_bytes)

    # Convert secret image to binary
    secret_binary = ''.join(format(pixel, '08b') for pixel in secret_pixels.flatten())

    # Combine dimension info and secret image data
    full_binary = dim_binary + secret_binary

    if len(full_binary) > cover_pixels.size:
        raise ValueError("Secret image too large for cover image")

    # Hide binary data in LSB of cover image
    flat_cover = cover_pixels.flatten()
    for i in range(len(full_binary)):
        flat_cover[i] = (flat_cover[i] & 0xFE) | int(full_binary[i])

    result_pixels = flat_cover.reshape(cover_pixels.shape)
    return Image.fromarray(result_pixels.astype(np.uint8))

def random_lsb_hide_image(cover_image, secret_image, key=None):
    """Hide image using randomized LSB positions"""
    cover_image = cover_image.convert('RGB')
    secret_image = secret_image.convert('RGB')

    # Setup random seed
    if key:
        seed = hash(key) % (2**32)
    else:
        seed = 12345

    # Resize secret image if needed
    cover_width, cover_height = cover_image.size
    secret_width, secret_height = secret_image.size

    max_secret_width = cover_width // 2
    max_secret_height = cover_height // 2

    if secret_width > max_secret_width or secret_height > max_secret_height:
        secret_image.thumbnail((max_secret_width, max_secret_height), Image.Resampling.LANCZOS)
        secret_width, secret_height = secret_image.size

    cover_pixels = np.array(cover_image)
    secret_pixels = np.array(secret_image)

    # Prepare binary data
    width_bytes = secret_width.to_bytes(4, 'big')
    height_bytes = secret_height.to_bytes(4, 'big')
    dim_binary = ''.join(format(byte, '08b') for byte in width_bytes + height_bytes)
    secret_binary = ''.join(format(pixel, '08b') for pixel in secret_pixels.flatten())
    full_binary = dim_binary + secret_binary

    flat_cover = cover_pixels.flatten()
    total_pixels = len(flat_cover)

    if len(full_binary) > total_pixels:
        raise ValueError("Secret image too large for cover image")

    # Generate random positions
    random.seed(seed)
    positions = random.sample(range(total_pixels), len(full_binary))

    # Hide data at random positions
    for i, pos in enumerate(positions):
        flat_cover[pos] = (flat_cover[pos] & 0xFE) | int(full_binary[i])

    result_pixels = flat_cover.reshape(cover_pixels.shape)
    return Image.fromarray(result_pixels.astype(np.uint8))

def extract_image_from_image(stego_image_path, key=None, algorithm='lsb_image'):
    """Extract hidden image using specified algorithm"""
    if algorithm == 'random_lsb':
        return random_lsb_extract_image(stego_image_path, key)
    else:  # Default LSB
        return lsb_extract_image(stego_image_path, key)

def lsb_extract_image(stego_image_path, key=None):
    """Extract hidden image using LSB technique"""
    stego_image = Image.open(stego_image_path)
    stego_pixels = np.array(stego_image).flatten()

    # Extract dimensions (first 64 bits)
    dim_bits = ''.join(str(stego_pixels[i] & 1) for i in range(64))

    width_bits = dim_bits[:32]
    height_bits = dim_bits[32:64]

    secret_width = int(width_bits, 2)
    secret_height = int(height_bits, 2)

    if secret_width <= 0 or secret_height <= 0:
        raise ValueError("Invalid image dimensions extracted")

    # Extract secret image data
    secret_size = secret_width * secret_height * 3  # RGB
    secret_bits_needed = secret_size * 8

    if 64 + secret_bits_needed > len(stego_pixels):
        raise ValueError("Not enough data to extract complete image")

    secret_bits = ''.join(str(stego_pixels[i + 64] & 1) for i in range(secret_bits_needed))

    # Convert bits back to pixels
    secret_pixels = []
    for i in range(0, len(secret_bits), 8):
        byte_bits = secret_bits[i:i+8]
        pixel_value = int(byte_bits, 2)
        secret_pixels.append(pixel_value)

    # Reshape to image
    secret_array = np.array(secret_pixels, dtype=np.uint8).reshape((secret_height, secret_width, 3))
    return Image.fromarray(secret_array)

def random_lsb_extract_image(stego_image_path, key=None):
    """Extract hidden image from randomized LSB positions"""
    stego_image = Image.open(stego_image_path)
    stego_pixels = np.array(stego_image).flatten()

    # Setup same random seed as hiding
    if key:
        seed = hash(key) % (2**32)
    else:
        seed = 12345

    random.seed(seed)
    total_pixels = len(stego_pixels)

    # We need to extract dimensions first, so we'll extract more bits initially
    # and determine the actual image size
    initial_positions = random.sample(range(total_pixels), min(10000, total_pixels))

    # Extract dimension bits (first 64 positions)
    dim_bits = ''.join(str(stego_pixels[initial_positions[i]] & 1) for i in range(64))

    width_bits = dim_bits[:32]
    height_bits = dim_bits[32:64]

    secret_width = int(width_bits, 2)
    secret_height = int(height_bits, 2)

    if secret_width <= 0 or secret_height <= 0:
        raise ValueError("Invalid image dimensions extracted")

    # Calculate total bits needed
    secret_size = secret_width * secret_height * 3  # RGB
    total_bits_needed = 64 + (secret_size * 8)  # dimensions + image data

    if total_bits_needed > len(initial_positions):
        # Generate more positions if needed
        random.seed(seed)
        positions = random.sample(range(total_pixels), total_bits_needed)
    else:
        positions = initial_positions[:total_bits_needed]

    # Extract secret image bits (skip first 64 for dimensions)
    secret_bits = ''.join(str(stego_pixels[positions[i + 64]] & 1) for i in range(secret_size * 8))

    # Convert bits back to pixels
    secret_pixels = []
    for i in range(0, len(secret_bits), 8):
        byte_bits = secret_bits[i:i+8]
        pixel_value = int(byte_bits, 2)
        secret_pixels.append(pixel_value)

    # Reshape to image
    secret_array = np.array(secret_pixels, dtype=np.uint8).reshape((secret_height, secret_width, 3))
    return Image.fromarray(secret_array)

def lsb_hide_message(image, message, key=None):
    """Original LSB hiding method"""
    image = image.convert('RGB')
    pixels = np.array(image)

    if not is_image_suitable(image):
        raise ValueError("Image is not suitable.")

    if key:
        key = hashlib.sha256(key.encode()).digest()
        message = encrypt_message(message, key)
    else:
        message = message.encode()

    packed_length = struct.pack('>I', len(message))
    full_message = packed_length + message
    binary_message = ''.join(format(byte, '08b') for byte in full_message)
    total_bits = len(binary_message)

    flat_pixels = pixels.flatten()
    if total_bits > len(flat_pixels):
        raise ValueError("Message too large for this image.")

    for i in range(total_bits):
        flat_pixels[i] = (flat_pixels[i] & 0xFE) | int(binary_message[i])

    new_pixels = flat_pixels.reshape(pixels.shape)
    return Image.fromarray(new_pixels)

def extract_message_lsb(image_path, key=None, algorithm='lsb'):
    """Extract message using specified algorithm"""
    if algorithm == 'dct':
        return dct_extract_message(image_path, key)

    elif algorithm == 'spread_spectrum':
        return spread_spectrum_extract_message(image_path, key)
    else:
        return lsb_extract_message(image_path, key)

def lsb_extract_message(image_path, key=None):
    """Original LSB extraction method"""
    image = Image.open(image_path)
    pixels = np.array(image).flatten()

    length_bits = ''.join(str(pixels[i] & 1) for i in range(32))
    message_length = struct.unpack('>I', int(length_bits, 2).to_bytes(4, 'big'))[0]
    total_bits = message_length * 8
    message_bits = ''.join(str(pixels[i + 32] & 1) for i in range(total_bits))
    byte_data = bytes(int(message_bits[i:i+8], 2) for i in range(0, len(message_bits), 8))

    if key:
        key = hashlib.sha256(key.encode()).digest()
        return decrypt_message(byte_data, key)
    else:
        return byte_data.decode()

def dct_extract_message(image_path, key=None):
    """Extract message from DCT coefficients"""
    image = Image.open(image_path)
    pixels = np.array(image, dtype=float)

    extracted_bits = []
    height, width = pixels.shape[:2]
    message_length = None

    # Process 8x8 blocks to collect all bits
    for channel in range(3):
        for i in range(0, height - 7, 8):
            for j in range(0, width - 7, 8):
                # Extract 8x8 block
                block = pixels[i:i+8, j:j+8, channel]

                # Apply DCT
                dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')

                # Extract bit from middle frequency coefficient
                # Match the embedding logic: positive means 1, negative means 0
                if dct_block[4, 4] >= 0:
                    extracted_bits.append('1')
                else:
                    extracted_bits.append('0')

                # Check if we have length and can determine total bits needed
                if len(extracted_bits) == 32 and message_length is None:
                    try:
                        length_bits = ''.join(extracted_bits[:32])
                        message_length = struct.unpack('>I', int(length_bits, 2).to_bytes(4, 'big'))[0]
                        if message_length <= 0 or message_length > 1000000:  # Sanity check
                            raise ValueError("Invalid message length")
                    except:
                        message_length = None
                        continue

                # Stop when we have all needed bits
                if message_length is not None and len(extracted_bits) >= 32 + message_length * 8:
                    break

            if message_length is not None and len(extracted_bits) >= 32 + message_length * 8:
                break

        if message_length is not None and len(extracted_bits) >= 32 + message_length * 8:
            break

    # Validate we have enough bits
    if len(extracted_bits) < 32:
        raise ValueError("Not enough data to extract message length")

    # Extract message length
    length_bits = ''.join(extracted_bits[:32])
    try:
        message_length = struct.unpack('>I', int(length_bits, 2).to_bytes(4, 'big'))[0]
    except:
        raise ValueError("Failed to decode message length")

    if message_length <= 0 or len(extracted_bits) < 32 + message_length * 8:
        raise ValueError(f"Invalid message length {message_length} or insufficient data")

    # Extract message bits
    message_bits = ''.join(extracted_bits[32:32 + message_length * 8])

    try:
        byte_data = bytes(int(message_bits[i:i+8], 2) for i in range(0, len(message_bits), 8))
    except:
        raise ValueError("Failed to decode message bits")

    if key:
        key = hashlib.sha256(key.encode()).digest()
        return decrypt_message(byte_data, key)
    else:
        return byte_data.decode()



def spread_spectrum_extract_message(image_path, key=None, spread_factor=16):
    """Extract message from spread spectrum embedding"""
    image = Image.open(image_path)
    pixels = np.array(image)

    if key:
        hash_key = hashlib.sha256(key.encode()).digest()
        master_seed = int.from_bytes(hash_key[:4], 'big')
    else:
        hash_key = None
        master_seed = 424242

    flat_pixels = pixels.flatten()
    total_pixels = len(flat_pixels)

    # Extract length first (32 bits)
    length_bits = []
    used_positions = set()

    for bit_index in range(32):
        # Use same bit-specific seed as embedding
        bit_seed = master_seed + bit_index * 1000
        random.seed(bit_seed)

        # Generate same positions as embedding
        positions = []
        attempts = 0
        while len(positions) < spread_factor and attempts < spread_factor * 3:
            pos = random.randint(0, total_pixels - 1)
            if pos not in used_positions:
                positions.append(pos)
                used_positions.add(pos)
            attempts += 1

        if len(positions) < spread_factor:
            random.seed(bit_seed)
            positions = [random.randint(0, total_pixels - 1) for _ in range(spread_factor)]

        # Vote on bit value
        ones = sum(1 for pos in positions if flat_pixels[pos] & 1)
        zeros = spread_factor - ones
        length_bits.append('1' if ones > zeros else '0')

    try:
        message_length = struct.unpack('>I', int(''.join(length_bits), 2).to_bytes(4, 'big'))[0]
    except:
        raise ValueError("Failed to decode message length")

    if message_length <= 0 or message_length > 100000:  # Sanity check
        raise ValueError(f"Invalid message length: {message_length}")

    # Extract message bits
    message_bits = []
    for bit_index in range(32, 32 + message_length * 8):
        bit_seed = master_seed + bit_index * 1000
        random.seed(bit_seed)

        positions = []
        attempts = 0
        while len(positions) < spread_factor and attempts < spread_factor * 3:
            pos = random.randint(0, total_pixels - 1)
            if pos not in used_positions:
                positions.append(pos)
                used_positions.add(pos)
            attempts += 1

        if len(positions) < spread_factor:
            random.seed(bit_seed)
            positions = [random.randint(0, total_pixels - 1) for _ in range(spread_factor)]

        # Vote on bit value
        ones = sum(1 for pos in positions if flat_pixels[pos] & 1)
        zeros = spread_factor - ones
        message_bits.append('1' if ones > zeros else '0')

    try:
        byte_data = bytes(int(''.join(message_bits)[i:i+8], 2) for i in range(0, len(''.join(message_bits)), 8))
    except:
        raise ValueError("Failed to decode message bits")

    if hash_key:
        return decrypt_message(byte_data, hash_key)
    else:
        return byte_data.decode()
