import cv2
import numpy as np

def get_dominant_color(image, k=3):
    """Get the most frequent dominant color in an image."""
    # Resize image to speed up the process
    resized_image = cv2.resize(image, (64, 64))
    # Reshape image to a list of pixels
    pixels = resized_image.reshape(-1, 3)
    # Convert to float32 for k-means
    pixels = np.float32(pixels)
    
    # Define criteria, number of clusters(K) and apply KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Get the most frequent color from clusters
    _, counts = np.unique(labels, return_counts=True)
    dominant_color = palette[np.argmax(counts)].astype(int)
    return dominant_color

def classify_color(color):
    """Classify color based on RGB values."""
    color_map = {
        'red': (7, 55, 187),
        'green': (9, 95, 52),
        'blue': (131, 89, 22),
        'yellow': (34, 169, 214),
        'orange': (12, 104, 199),
        'purple': (75, 36, 100)
    }
    color_name = 'unknown'
    min_dist = float('inf')
    for name, rgb in color_map.items():
        dist = np.linalg.norm(np.array(rgb) - color)
        if dist < min_dist:
            min_dist = dist
            color_name = name
    return color_name

def process_card(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Define center crop region (focus only on the very center)
    h, w, _ = image.shape
    center_crop = image[int(h*0.45):int(h*0.55), int(w*0.45):int(w*0.55)]  # Narrow center region
    
    # Get the dominant color in the cropped image
    dominant_color = get_dominant_color(center_crop)
    print(f"Dominant color for {image_path}: {dominant_color}")  # Debug print
    
    # Classify the color to distinguish the card
    color_name = classify_color(dominant_color)
    print(f"Classified color for {image_path}: {color_name}")  # Debug print
    
    return color_name

# List of image paths for each card
image_paths = [
    "KakaoTalk_20241111_203433453_01.jpg",
    "KakaoTalk_20241111_203433453_02.jpg",
    "KakaoTalk_20241111_203433453_03.jpg",
    "KakaoTalk_20241111_203433453_04.jpg",
    "KakaoTalk_20241111_203433453_05.jpg",
    "KakaoTalk_20241111_203433453.jpg"
]

# Process each card
card_colors = {}
for idx, path in enumerate(image_paths, 1):
    color_name = process_card(path)
    card_colors[f"Card {idx}"] = color_name

print(card_colors)
