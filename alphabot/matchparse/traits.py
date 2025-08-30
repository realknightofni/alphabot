import math

import cv2
import numpy as np

from collections import defaultdict

# in RGB, don't put 255, use 254 instead
TRAIT_COLORS = {
    'blue': (55, 83, 254),  # blue
    'red': (246, 54, 86),  # red
    'yellow': (254, 221, 65),  # yellow
    'green': (220, 254, 65)  # lime green
}


# TOOD: this should scale with some reference object
MIN_RADIUS = 10
MAX_RADIUS = 40

CIRCULARITY_THRESHOLD = 0.7

# TODO: might want to consildate this code into base.py or something
def get_center(x, y, w, h):
    return int(x + w/2), int(y+h/2)


def calculate_spacing(boxes):
    if len(boxes) < 2:
        # TODO: this condition is not enough.
        # we need to make sure there's at least 2 boxes in each direction
        raise ValueError(f'Could not determine box spacing with {len(boxes)} boxes')

    centers = [get_center(*b) for b in boxes]

    # Calculate horizontal spacing
    x_coords = sorted(set(x for x, _ in centers))
    horizontal_spacing = np.median(np.diff(x_coords))

    # Calculate vertical spacing
    y_coords = sorted(set(y for _, y in centers))
    vertical_spacing = np.median(np.diff(y_coords))

    return math.ceil(horizontal_spacing), math.ceil(vertical_spacing)


def get_average_color(image):
    # Calculate the average color of the image
    average_color = np.mean(image, axis=(0, 1))
    return average_color


def closest_color(image, colors=TRAIT_COLORS):
    # Find the closest color from the predefined colors
    min_distance = float('inf')
    average_color = get_average_color(image)

    closest_color_name = None
    for color_name, color_rgb in colors.items():
        distance = np.linalg.norm(average_color - color_rgb)
        if distance < min_distance:
            min_distance = distance
            closest_color_name = color_name

    return closest_color_name


def get_trait_and_missing_bboxes(image):
    """Use the missing trait bounding boxes to build the trait bounding boxes."""
    missing_bboxes = get_trait_missing_bboxes(image)
    grid_cols = 6
    grid_rows = 8

    horizontal_spacing, vertical_spacing = calculate_spacing(missing_bboxes)
    trait_width = math.ceil(np.median([b[2] for b in missing_bboxes]))
    trait_height = math.ceil(np.median([b[3] for b in missing_bboxes]))

    max_x = int(max([box[0] for box in missing_bboxes]))
    max_y = int(max([box[1] for box in missing_bboxes]))
    min_x = int(max_x - 5*horizontal_spacing)
    min_y = int(max_y - 7*vertical_spacing)

    trait_bboxes = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = int(min_x + col * horizontal_spacing)
            y = int(min_y + row * vertical_spacing)

            is_detected = any(
                abs(x - box[0]) < 15 and abs(y - box[1]) < 15  # Tolerance for position
                for box in missing_bboxes
            )

            if not is_detected:
                trait_bboxes.append((x, y, trait_width, trait_height))

    return trait_bboxes, missing_bboxes


# TODO: consolidate into a qa.py file or something
def highlight_and_save_contours(image, contours, filename):
    image_copy = image.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 1)
    cv2.imwrite(filename, image_copy)


def get_trait_missing_bboxes(image):
    """Find the bounding boxes for (?) missing traits."""
    # HACK TO GET RID OF THE BIG BOX AROUND THE MATCH.
    # NOTE: FIX THIS JANK HACK
    # this just chops off the right side of the image
    _, width = image.shape[:2]
    x_min = width//4
    left_side = image[:, :width-width//4]

    hsv_image = cv2.cvtColor(left_side, cv2.COLOR_BGR2HSV)
    # cv2.imwrite('testtraits_hcsv.png', hsv_image)

    lower_gray = np.array([0, 0, 90])   # Lower bound for gray
    upper_gray = np.array([50, 50, 175])  # Upper bound for gray

    mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
    # cv2.imwrite('testtraits.png', mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    minsize_contours = []
    for contour in contours:
        _, radius = cv2.minEnclosingCircle(contour)

        if MIN_RADIUS <= radius <= MAX_RADIUS:
            minsize_contours.append(contour)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0

            if circularity >= CIRCULARITY_THRESHOLD:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x,y,w,h))
    
    # highlight_and_save_contours(image, contours, 'testtraits_allcontours.png')
    # highlight_and_save_contours(image, minsize_contours, 'testtraits_minsize.png')

    # Drop any semi-large circles
    # No third artifact uses the same symbol
    # so we need to drop false negatives
    median_width = np.median([b[2] for b in bboxes])
    bboxes2 = [b for b in bboxes if abs(b[2]-median_width) < 0.1 * median_width]
    # print(f'{len(bboxes)} >> {len(bboxes2)}')
    return bboxes2


def mask_percent(mask):
    total_pixels = mask.size
    white_pixels = np.sum(mask==255)

    return white_pixels / total_pixels


def apply_mask(image, rgb_color, tolerance=20):
    r, g, b = rgb_color

    lower = np.array([max(0, b - tolerance),
                      max(0, g - tolerance),
                      max(0, r - tolerance)])
    
    upper = np.array([min(255, b + tolerance),
                      min(255, g + tolerance),
                      min(255, r + tolerance)])
    
    mask = cv2.inRange(image.copy(), lower, upper)
    return mask


def determine_primary_bbox_color(image, bbox=None):
    if bbox is None:
        h, w, _ = image.shape
        x, y = 0, 0
    else:
        x, y, w, h = bbox

    color2percent = {}
    for color_name, rgb_color in TRAIT_COLORS.items():
        icon = image[y:y+h, x:x+w]
        mask = apply_mask(icon, rgb_color)
        
        color2percent[color_name] = mask_percent(mask)
    
    primary_color = max(color2percent, key=lambda k:color2percent[k])
    perc_color = color2percent[primary_color]
    # print(perc_color)

    return primary_color, perc_color


def split_trait_bboxes_by_color(image, trait_bboxes):
    color2bboxes = defaultdict(list)
    for bbox in trait_bboxes:
        # TODO: might want to ignore bboxes where the primary color
        # isn't a certain percentage
        color, _ = determine_primary_bbox_color(image, bbox)

        color2bboxes[color].append(bbox)

    return color2bboxes
