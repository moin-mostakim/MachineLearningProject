"""
Synthetic Retinal Data Generator for Multi-Task DR Learning.

Generates synthetic fundus images with:
- Realistic fundus appearance (vessels, optic disc, macula)
- Lesion regions (MA, HE, SE, EX) with ground truth masks
- DR severity grades correlated with lesion counts/areas

Based on characteristics described in:
- Porwal et al. (2018) IDRiD Challenge
- Decencière et al. (2014) Feedback on a publicly distributed image database: the Messidor database
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_fundus_background(size=512):
    """Create a realistic fundus background with gradient coloring."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    
    # Create circular fundus mask
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 2 - 10
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    fundus_mask = dist < radius
    
    # Base retinal color (reddish-orange)
    base_color = np.array([0.75, 0.35, 0.15])
    
    # Add radial gradient (darker at edges)
    gradient = 1 - (dist / radius) * 0.3
    gradient = np.clip(gradient, 0, 1)
    
    for c in range(3):
        img[:, :, c] = base_color[c] * gradient * fundus_mask
    
    # Add some texture noise
    noise = np.random.normal(0, 0.02, (size, size, 3))
    img = np.clip(img + noise * fundus_mask[:, :, np.newaxis], 0, 1)
    
    return img, fundus_mask


def add_optic_disc(img, size=512):
    """Add optic disc to fundus image."""
    # Optic disc location (typically nasal side, offset from center)
    od_x = size // 2 + size // 5
    od_y = size // 2
    od_radius = size // 12
    
    y, x = np.ogrid[:size, :size]
    od_dist = np.sqrt((x - od_x)**2 + (y - od_y)**2)
    od_mask = od_dist < od_radius
    
    # Optic disc is bright yellowish
    od_color = np.array([0.95, 0.9, 0.7])
    fade = np.clip(1 - od_dist / od_radius, 0, 1)
    
    for c in range(3):
        img[:, :, c] = np.where(od_mask, 
                                img[:, :, c] * (1 - fade) + od_color[c] * fade,
                                img[:, :, c])
    return img


def add_macula(img, size=512):
    """Add macula (darker region) to fundus image."""
    # Macula location (temporal to optic disc)
    mac_x = size // 2 - size // 6
    mac_y = size // 2
    mac_radius = size // 10
    
    y, x = np.ogrid[:size, :size]
    mac_dist = np.sqrt((x - mac_x)**2 + (y - mac_y)**2)
    mac_mask = mac_dist < mac_radius
    
    # Macula is slightly darker
    fade = np.clip(1 - mac_dist / mac_radius, 0, 1) * 0.15
    
    for c in range(3):
        img[:, :, c] = np.where(mac_mask, 
                                img[:, :, c] * (1 - fade),
                                img[:, :, c])
    return img


def add_vessels(img, fundus_mask, size=512, num_vessels=8):
    """Add blood vessel structures to fundus image."""
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    
    # Origin from optic disc
    origin_x = size // 2 + size // 5
    origin_y = size // 2
    
    for _ in range(num_vessels):
        # Random vessel path
        angle = np.random.uniform(-np.pi, np.pi)
        length = np.random.uniform(size // 3, size // 2)
        
        points = [(origin_x, origin_y)]
        x, y = origin_x, origin_y
        
        for _ in range(10):
            # Add some curvature
            angle += np.random.normal(0, 0.2)
            step = length / 10
            x += step * np.cos(angle)
            y += step * np.sin(angle)
            
            if 0 < x < size and 0 < y < size:
                points.append((int(x), int(y)))
        
        # Draw vessel (dark red)
        if len(points) > 1:
            width = np.random.randint(2, 5)
            draw.line(points, fill=(120, 30, 30), width=width)
    
    # Apply slight blur for realism
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return np.array(pil_img).astype(np.float32) / 255


def generate_lesion_mask(size, lesion_type, count, fundus_mask):
    """Generate lesion mask for a specific lesion type."""
    mask = np.zeros((size, size), dtype=np.float32)
    
    # Lesion characteristics based on type
    if lesion_type == 0:  # Microaneurysms (MA) - small red dots
        radius_range = (2, 5)
    elif lesion_type == 1:  # Hemorrhages (HE) - larger red blots
        radius_range = (5, 20)
    elif lesion_type == 2:  # Soft Exudates (SE) - fluffy white regions
        radius_range = (10, 25)
    else:  # Hard Exudates (EX) - small bright yellow deposits
        radius_range = (3, 12)
    
    valid_positions = np.where(fundus_mask)
    if len(valid_positions[0]) == 0:
        return mask
    
    for _ in range(count):
        idx = np.random.randint(len(valid_positions[0]))
        cy = valid_positions[0][idx]
        cx = valid_positions[1][idx]
        
        radius = np.random.uniform(*radius_range)
        
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        if lesion_type == 2:  # Soft exudates have irregular edges
            lesion = (dist < radius * (1 + 0.3 * np.random.randn())).astype(np.float32)
        else:
            lesion = (dist < radius).astype(np.float32)
        
        mask = np.maximum(mask, lesion)
    
    return mask * fundus_mask


def add_lesions_to_image(img, masks, size=512):
    """Add visual lesion appearance to image based on masks."""
    result = img.copy()
    
    lesion_colors = [
        np.array([0.6, 0.1, 0.1]),   # MA - dark red
        np.array([0.5, 0.05, 0.05]), # HE - darker red
        np.array([0.95, 0.95, 0.8]), # SE - pale/white
        np.array([0.95, 0.9, 0.4]),  # EX - yellow
    ]
    
    for i, (mask, color) in enumerate(zip(masks, lesion_colors)):
        for c in range(3):
            result[:, :, c] = np.where(mask > 0.5,
                                       result[:, :, c] * 0.3 + color[c] * 0.7,
                                       result[:, :, c])
    
    return result


def determine_dr_grade(lesion_counts):
    """Determine DR grade based on lesion counts (simplified grading)."""
    ma_count, he_count, se_count, ex_count = lesion_counts
    total_lesions = sum(lesion_counts)
    
    if total_lesions == 0:
        return 0  # No DR
    elif total_lesions < 5 and he_count == 0:
        return 1  # Mild
    elif total_lesions < 15 and se_count < 3:
        return 2  # Moderate
    elif total_lesions < 30:
        return 3  # Severe
    else:
        return 4  # Proliferative


def generate_sample(idx, size=512):
    """Generate a single synthetic fundus sample with masks and grade."""
    np.random.seed(config.SEED + idx)
    
    # Create fundus background
    img, fundus_mask = create_fundus_background(size)
    img = add_optic_disc(img, size)
    img = add_macula(img, size)
    img = add_vessels(img, fundus_mask, size)
    
    # Determine lesion counts based on random severity
    severity = np.random.choice([0, 1, 2, 3, 4], p=[0.25, 0.2, 0.25, 0.2, 0.1])
    
    if severity == 0:
        lesion_counts = [0, 0, 0, 0]
    elif severity == 1:
        lesion_counts = [np.random.randint(1, 4), 0, 0, np.random.randint(0, 2)]
    elif severity == 2:
        lesion_counts = [np.random.randint(3, 8), np.random.randint(1, 3), 
                        np.random.randint(0, 2), np.random.randint(1, 5)]
    elif severity == 3:
        lesion_counts = [np.random.randint(5, 15), np.random.randint(3, 8),
                        np.random.randint(1, 4), np.random.randint(3, 10)]
    else:
        lesion_counts = [np.random.randint(10, 25), np.random.randint(5, 15),
                        np.random.randint(3, 8), np.random.randint(8, 20)]
    
    # Generate lesion masks
    masks = []
    for lesion_type, count in enumerate(lesion_counts):
        mask = generate_lesion_mask(size, lesion_type, count, fundus_mask)
        masks.append(mask)
    
    # Add lesions to image
    img = add_lesions_to_image(img, masks, size)
    
    # Determine grade
    grade = determine_dr_grade(lesion_counts)
    
    # Stack masks: (4, H, W)
    combined_mask = np.stack(masks, axis=0)
    
    return img, combined_mask, grade


def generate_dataset(num_samples, output_dir, split_name):
    """Generate complete dataset split."""
    images_dir = os.path.join(output_dir, split_name, "images")
    masks_dir = os.path.join(output_dir, split_name, "masks")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    labels = []
    
    for i in range(num_samples):
        img, mask, grade = generate_sample(i + (0 if split_name == "train" else 1000 if split_name == "val" else 2000))
        
        # Save image
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(os.path.join(images_dir, f"{i:04d}.png"))
        
        # Save masks (as separate files per lesion type)
        for j, lesion_name in enumerate(config.LESION_NAMES):
            mask_uint8 = (mask[j] * 255).astype(np.uint8)
            mask_path = os.path.join(masks_dir, f"{i:04d}_{lesion_name}.png")
            Image.fromarray(mask_uint8).save(mask_path)
        
        labels.append(grade)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_samples} {split_name} samples")
    
    # Save labels
    labels_path = os.path.join(output_dir, split_name, "labels.txt")
    with open(labels_path, 'w') as f:
        for i, label in enumerate(labels):
            f.write(f"{i:04d},{label}\n")
    
    return labels


def create_synthetic_dataset():
    """Create complete synthetic IDRiD-like dataset."""
    eff_config = config.get_effective_config()
    
    print(f"Generating synthetic dataset (DEBUG_MODE={config.DEBUG_MODE})")
    print(f"Train: {eff_config['num_train_samples']}, Val: {eff_config['num_val_samples']}, Test: {eff_config['num_test_samples']}")
    
    data_dir = config.DATA_DIR
    
    train_labels = generate_dataset(eff_config['num_train_samples'], data_dir, "train")
    val_labels = generate_dataset(eff_config['num_val_samples'], data_dir, "val")
    test_labels = generate_dataset(eff_config['num_test_samples'], data_dir, "test")
    
    # Print label distribution
    for split_name, labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        print(f"\n{split_name} label distribution:")
        for grade in range(5):
            count = labels.count(grade)
            print(f"  Grade {grade} ({config.CLASS_NAMES[grade]}): {count}")
    
    print(f"\nDataset saved to {data_dir}")
    return data_dir


if __name__ == "__main__":
    create_synthetic_dataset()
