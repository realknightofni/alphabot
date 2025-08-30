"""Helper functions for genres.

Genre bboxes are inferred from position of the artifact bboxes.

"""
import logging
import math

import cv2
import numpy as np

from . import hashes

from collections import Counter
from functools import lru_cache

logger = logging.getLogger(__name__)

EXPECTED_GENRES = 8

# factors relative to the min/max of the artifact bboxes
FACTOR_SPACING_INITIAL = 0.0595744
FACTOR_WIDTH = 0.2510638
#FACTOR_SPACING_GAPS = 0.042553191489
FACTOR_SPACING_GAPS = 0.043

PADDING = 3

GENRES = ['Crit', 'Evasion', 'Frost', 'Heal',
          'Health', 'Innerfire', 'Mech', 'Shield',
          'Spell', 'Toxin', 'Vulnerable', 'Weaponry'
          ]

GENRE2CODE = {'Crit': 'CR',
              'Evasion': 'EV',
              'Frost': 'FR',
              'Heal': 'HL',
              'Health': 'HP',
              'Innerfire': 'IF',
              'Mech': 'ME',
              'Shield': 'SH',
              'Spell': 'SP',
              'Toxin': 'TX',
              'Vulnerable': 'VU',
              'Weaponry': 'WP',
              }


def get_genre_code(genre_name):
    return GENRE2CODE.get(genre_name, '??')


# map if genre level to min xp
GENRE_TIERS = {1: 4,
               2: 12,
               3: 24,
               4: 40
               }


def get_level_from_exp(exp):
    """Return the expected level (0-4) given the experience # (0-40+)."""
    if exp == -1 or exp == None:
        return None
    if exp < 0:
        return 0
    for level, min_exp in sorted(GENRE_TIERS.items(), reverse=True):
        if exp >= min_exp:
            return level
    return 0

@lru_cache
def get_exp_range_from_level(level):
    max_level = max(GENRE_TIERS.keys())
    if level > max_level:
        return 0, 9999

    if level == 0:
        min_exp = 0
    else:
        min_exp = GENRE_TIERS[level]
    
    if level == max_level:
        max_exp = 9999
    else:
        max_exp = GENRE_TIERS[level+1]

    return min_exp, max_exp


def calculate_reference_width(all_artifact_bboxes):
    min_x = min(b[0] for b in all_artifact_bboxes)
    base_x = max(b[0]+b[2] for b in all_artifact_bboxes)

    reference_width = (base_x - min_x) 

    return reference_width, base_x


def infer_genre_bboxes(player_artifact_bboxes, reference_width, base_x):
    if len(player_artifact_bboxes) == 0:
        return []

    if len(player_artifact_bboxes) > 3:
        raise NotImplementedError(f'Logic currently only supports the 3 artifact bounding boxes, but received: {len(player_artifact_bboxes)}')

    y = min(b[1] for b in player_artifact_bboxes)
    h = max(b[3] for b in player_artifact_bboxes)

    # all floats, need to floor/ceil later
    fs0 = reference_width * FACTOR_SPACING_INITIAL
    fw = reference_width * FACTOR_WIDTH
    fgap = reference_width * FACTOR_SPACING_GAPS

    genre_bboxes = []
    for i in range(EXPECTED_GENRES):
        x = base_x + fs0 + i*(fw + fgap)

        x = math.floor(x) - PADDING
        w = math.ceil(fw) + PADDING
        bbox = (x, y, w, h)
        genre_bboxes.append(bbox)

    return genre_bboxes


def detect_yellow_bboxes(image, min_area_perc=0.005, bottom_region_ratio=0.4):
    # Convert to HSV color space (better for color detection)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #cv2.imwrite('genrehsv.png', hsv)

    # Define range for yellow colors in HSV
    lower_yellow = np.array([20, 70, 150])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for yellow regions
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    #cv2.imwrite('genremask.png', mask)

    # remove some background noise
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    #cv2.imwrite('genremask2.png', mask)

    # Focus only on the bottom region of the image
    height = image.shape[0]
    bottom_start = int(height * (1 - bottom_region_ratio))
    bottom_region = mask[bottom_start:height, :]
    _h, _w = bottom_region.shape[:2]
    bottom_area = _h * _w
    min_area = int(min_area_perc * bottom_area)

    # Find contours in the bottom region
    contours, _ = cv2.findContours(bottom_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by area and adjust their coordinates (since we cropped the region)
    yellow_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            # Adjust contour coordinates to full image space
            cnt_adjusted = cnt + [0, bottom_start]

            # Check aspect ratio
            x, y, w, h = cv2.boundingRect(cnt_adjusted)
            aspect_ratio = float(w)/h

            if aspect_ratio > 1.8:
                # "hack" mostly in place to catch when shield icons
                # merge the second and third star
                # Left half
                left_half = cnt_adjusted.copy()
                left_half[:, :, 0] = np.clip(left_half[:, :, 0], x, x + w//2)

                # Right half
                right_half = cnt_adjusted.copy()
                right_half[:, :, 0] = np.clip(right_half[:, :, 0], x + w//2, x + w)

                yellow_contours.extend([left_half, right_half])
            else:
                yellow_contours.append(cnt_adjusted)
    
    if len(yellow_contours) == 0:
        return yellow_contours

    # remove contours less than 50% of the max width
    bboxes = [cv2.boundingRect(c) for c in yellow_contours]
    widths = [w for _,_,w,_  in bboxes]
    max_width = max(widths)
    perc_diffs = [abs(w-max_width)/max_width for w in widths]
    yellow_contours = [yc for yc, diff in zip(yellow_contours, perc_diffs) if diff <= 0.35]

    # if len(contours) > len(yellow_contours):
    #     import time
    #     time.sleep(30)
    #     print(f'{len(contours)} - {len(yellow_contours)} - {areas}')
    #     print('sleeping 30')

    return yellow_contours


def get_genre_level(image, bbox):
    x, y, w, h = bbox
    icon = image[y:y+h, x:x+w]
    yellow_contours = detect_yellow_bboxes(icon)

    # Adjust contour coordinates to be relative to original image
    adjusted_contours = []
    for contour in yellow_contours:
        # Shift contour points by (x, y) offset
        adjusted_contour = contour + np.array([x, y])
        adjusted_contours.append(adjusted_contour)

    return len(yellow_contours), adjusted_contours


def get_top_right(image):
    height, width = image.shape[:2]

    top_right = image[:height//2, width//2:]
    return top_right


def read_genre_exp(image, bbox, reader):
    x, y, w, h = bbox
    icon = image[y:y+h, x:x+w]
    top_right = get_top_right(icon)

    gray = cv2.cvtColor(top_right, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)

    texts = reader.readtext(blurred, allowlist='0123456789')
    if len(texts) > 1:
        logger.warning('More than one number (%s) found in the genre icon.', len(texts))
        return -1, 0

    if not texts:
        return -1, 0

    _, num, conf = texts[0][:3]
    genre_exp = int(num) if num.isnumeric() else -1
    return genre_exp, conf


def to_genre_shorthand(genre_name, genre_level, genre_exp):
    genre_code = get_genre_code(genre_name)
    return f'{genre_code}{genre_level}_{genre_exp:02d}'


def infer_main_genres(players, min_count=3):
    player_genres = []
    for player in players:
        genre_names = frozenset([guess.name for guess in player.genre_guesses if hashes.is_known(guess)])
        player_genres.append(genre_names)

    genre_counts = Counter(player_genres)
    print(genre_counts)
    most_common_game_genre, _ = genre_counts.most_common(1)[0]
    is_above_threshold = genre_counts[most_common_game_genre] >= min_count
    is_full_8 = len(most_common_game_genre) == 8

    if is_full_8 & is_above_threshold:
        return sorted(most_common_game_genre)
    else:
        print(f'Could not determine main genres: {genre_counts}')
        return []


def _is_valid_main_genres(main_genres):
    if len(main_genres) != 8:
        return False
    
    return True


def get_banned_genres(main_genres):
    if not _is_valid_main_genres(main_genres):
        return []
    
    return sorted(set(GENRE2CODE.keys()) - set(main_genres))


def fix_genre_guesses(player, main_genres):
    """Fix genres (in place) given the main genres."""
    if not _is_valid_main_genres(main_genres):
        return player.genre_guesses

    current_genre_names = [guess.name for guess in player.genre_guesses]
    num_unknown = current_genre_names.count('UNKNOWN_GENRE')
    if num_unknown > 1:
        return player.genre_guesses

    expected_genres = set(main_genres) - set(current_genre_names)
    if len(expected_genres) > 1:
        return player.genre_guesses
    if len(expected_genres) == 0:
        return player.genre_guesses

    expected_genre_name, = expected_genres
    bad_genres = set(current_genre_names) - set(main_genres)

    if bad_genres:
        bad_genre_name = list(bad_genres)[0]
        bad_genre_i = current_genre_names.index(bad_genre_name)
    else:
        # there's a duplicate
        # find the highest hamming and label that as the bad index
        counts = Counter(current_genre_names)

        most_common_genre, cnt = counts.most_common(1)[0]
        if cnt > 2:
            print(f'Multiple ({cnt}) genre guesses with: {most_common_genre}')
            return player.genre_guesses

        genre_idxs = [i for i, genre in enumerate(current_genre_names) if genre == most_common_genre]
        hammings = [player.genre_hammings[i] for i in genre_idxs]
        if set(hammings) == 1:
            print('All genre indexes have the same hamming. Cannot determine which one to replace.')
            return player.genre_guesses

        # NOTE: currently arbitary if there is a tie in hammings
        bad_genre_name = most_common_genre
        bad_genre_i = genre_idxs[np.argmax(hammings)]

    expected_genre = hashes.Genre(expected_genre_name, '_')
    fixed_genres = []
    for i, genre in enumerate(player.genre_guesses):
        if i == bad_genre_i:
            fixed_genres.append(expected_genre)
        else:
            fixed_genres.append(genre)

    return fixed_genres
    # player.genre_guesses[bad_genre_i] = expected_genre
    # logger.info('Changed player (%s) genre from %s to %s',
    #             player.name,
    #             bad_genre_name,
    #             expected_genre_name)


def agrees_with_neighors(value, i, existing_elements):
    if i == 0:
        right = existing_elements[1]
        if right == -1:
            return True
        else:
            return value >= existing_elements[1]
    elif i == len(existing_elements) - 1:
        left = existing_elements[-2]
        if left == -1:
            return True
        else:
            return existing_elements[-2] >= value
    else:
        # left = existing_elements[i-1]
        # right = existing_elements[i+1]
        left = find_non_negative(existing_elements, i, 'left')
        right = find_non_negative(existing_elements, i, 'right')

        if left is None and right is None:
            return  True
        elif left is None:
            return value >= right
        elif right is None:
            return left >= value
        else:
            return left >= value >= right


def fix_genre_levels(original_levels, original_exps):
    """Fix genre level guesses based off knowing they should be in descending order.
    
    Also takes into account the expected level from the exp number.

    """
    num_elements = len(original_levels)
    changed = [False] * num_elements
    if num_elements <= 2:
        return original_levels, changed

    fixed_levels = list(original_levels)
    changed = [False] * num_elements

    # fix middle elements
    for i in range(1, len(original_levels) - 1):
        left = original_levels[i - 1]
        curr = original_levels[i]
        right = original_levels[i + 1]

        if left == right and left != curr:
            fixed_levels[i] = left
            changed[i] = True

    # fix start element
    if original_levels[0] < fixed_levels[1]:
        fixed_levels[0] = infer_genre_level(original_exps[0], right=fixed_levels[1])
        changed[0] = True

    # fix last element
    if fixed_levels[-2] < original_levels[-1]:
        fixed_levels[-1] = infer_genre_level(original_exps[-1], left=fixed_levels[-2])
        changed[0] = True

    second_pass_levels = list(fixed_levels)
    # second pass through middle elements to check desc order
    for i in range(1, len(fixed_levels) - 1):
        left = fixed_levels[i - 1]
        curr = fixed_levels[i]
        right = fixed_levels[i + 1]

        if left > right and (left < curr or curr < right):
            exp = original_exps[i]
            second_pass_levels[i] = infer_genre_level(exp, left, right)
            changed[i] = True

    # after all the initial passes, check the first element again
    if original_exps[0] != -1:
        first_exp_level = get_level_from_exp(original_exps[0])
        if first_exp_level != second_pass_levels[0] and first_exp_level >= second_pass_levels[0]:
            second_pass_levels[0] = first_exp_level
            changed[i] = True

    # consider exp >=30 to be "accurate"
    # 20-29 might not be "accurate"
    # b/c the health icon sometimes gets interpreted as a 2
    thresholds = [100, 100, 60, 60, 50, 50, 50, 50]  # TODO: same as the fix exp function. move this to a global
    for i in range(num_elements):
        exp = original_exps[i]
        lvl = second_pass_levels[i]
        lvl_from_exp = get_level_from_exp(exp)
        if (thresholds[i] >= exp >= 30 and 
            lvl != lvl_from_exp and 
            agrees_with_neighors(lvl_from_exp, i, second_pass_levels)):
            second_pass_levels[i] = lvl_from_exp
            changed[i] = True

    if any(changed):
        print(f'Changed from {original_levels} >> {second_pass_levels}')

    return second_pass_levels, changed


def infer_genre_level(exp, left=None, right=None):
    exp_level = get_level_from_exp(exp)

    if exp_level is not None:
        if left is None:
            is_valid_exp = exp_level >= right
        elif right is None:
            is_valid_exp = left >= exp_level
        else:
            is_valid_exp = left >= exp_level >= right

        if is_valid_exp:
            return exp_level

    if left is None:
        return right
    elif right is None:
        return left
    else:
        # lean towards the smaller number
        # anecdotally, the bottom right of an image has more issues
        return right


def infer_genre_exp(original_exp, level, left=None, right=None):
    min_exp, max_exp = get_exp_range_from_level(level)

    # handle -1 cases
    if right == -1:
        return min_exp
    if left == -1:
        if right is None:
            return min_exp
        return max(min_exp, right)

    if left is None and right is None:
        return original_exp
    elif left is None:
        return _infer_genre_exp_first(level, right)
    elif right is None:
        return _infer_genre_exp_last(level, left)
    else:
        return _infer_genre_exp_middle(original_exp, level, left, right)


def _infer_genre_exp_first(level, right):
    return right


def _infer_genre_exp_last(level, left):
    min_exp, _ = get_exp_range_from_level(level)
    if left is None:
        return min_exp
    else:
        return min(min_exp, left)


def _infer_genre_exp_middle(original_exp, level, left, right):
    def _is_valid(num, valid_min, valid_max):
        return valid_min <= num <= valid_max
    min_exp, max_exp = get_exp_range_from_level(level)

    if left == right:
        return left

    if min_exp > left:
        return right
    if right > max_exp:
        return right

    valid_min = max(min_exp, right)
    valid_max = min(max_exp, left)

    if valid_min == valid_max:
        return valid_min

    # TODO: remove this "hack"
    # anecdotally, these numbers aren't picked up that well
    # by easy_ocr. intentionally ordered
    # apologies for the voodoo magic
    HACK_NUMBERS = [7, 9, 8, 6, 5, 4, 3]
    for hack_num in HACK_NUMBERS:
        if _is_valid(hack_num, valid_min, valid_max):
            return hack_num

    # TODO: revisit this one day
    return valid_min


def fix_genre_exps(original_exps, fixed_levels, original_guesses):
    num_elements = len(original_exps) 
    changed = [False] * num_elements
    if num_elements <= 2:
        return original_exps, changed

    fixed_exps = list(original_exps)

    # level/exp mismatch
    # check if using the ones digit works
    for i in range(num_elements):
        curr = fixed_exps[i]
        if curr == -1:
            continue
        level = fixed_levels[i]
        min_exp, max_exp = get_exp_range_from_level(level)

        is_level0_1 = level <= 1
        is_double_digit = curr >= 10
        if is_level0_1 and is_double_digit:
                single_digit = curr % 10
                if max_exp >= single_digit >= min_exp and agrees_with_neighors(single_digit, i, fixed_exps):
                    print(f'SINGLE DIGIT CHANGE: {single_digit}')
                    fixed_exps[i] = single_digit
                    changed[i] = True

    # fix values that are clearly wrong
    thresholds = [100, 100, 60, 60, 50, 50, 50, 50]
    for i, curr in enumerate(original_exps):
        threshold = thresholds[i]
        if curr > threshold:
            level = fixed_levels[i]
            left = find_non_negative(fixed_exps, i, 'left')
            right = find_non_negative(fixed_exps, i, 'right')
            # left = original_exps[i-1] if i != 0 else None
            # right = original_exps[i+1] if i != num_elements - 1 else None
            fixed_exps[i] = infer_genre_exp(curr, level, left, right)
            changed[i] = True

    # level/exp mismatch
    # check if +10 works
    for i in range(num_elements):
        curr = original_exps[i]
        if curr == -1:
            continue
        level = fixed_levels[i]
        min_exp, max_exp = get_exp_range_from_level(level)

        is_single_digit = curr < 10
        is_mismatch = curr < min_exp
        fixed_curr = curr + 10
        is_agree = agrees_with_neighors(fixed_curr, i, original_exps)
        if is_single_digit and is_mismatch and is_agree:
            fixed_exps[i] = curr + 10
            changed[i] = True

    # fix isolated -1s in the middle elements
    for i in range(1, num_elements - 1):
        left, curr, right = fixed_exps[i-1:i+2]
        if curr == -1 and left != -1 and right != -1:
            level = fixed_levels[i]
            fixed_exps[i] = infer_genre_exp(curr, level, left, right)
            changed[i] = True

    # fix first and last element if -1
    if fixed_exps[0] == -1:
        fixed_exps[0] = infer_genre_exp(fixed_exps[0], fixed_levels[0], None, fixed_exps[1])
        changed[0] = True

    if fixed_exps[-1] == -1:
        fixed_exps[-1] = infer_genre_exp(fixed_exps[-1], fixed_levels[-1], fixed_exps[-2], None)
        changed[-1] = True

    # fix pair of -1s
    for i in range(1, num_elements - 2):
        # use fixes from above
        left = fixed_exps[i-1]
        curr0 = fixed_exps[i]
        curr1 = fixed_exps[i+1]
        right = fixed_exps[i+2]

        is_both_bad = curr0 == -1 and curr1 == -1 
        is_outside_good = left != -1 and right != -1
        if is_both_bad and is_outside_good:
            fixed_exps[i] = infer_genre_exp(curr0, fixed_levels[i], left, right)
            changed[i] = True

            fixed_exps[i+1] = infer_genre_exp(curr1, fixed_levels[i+1], fixed_exps[i], right)
            changed[i+1] = True

    # For any unchanged values
    # Fix disagreement with neighbors
    for i in range(1, num_elements - 1):
        value = fixed_exps[i]
        if value == -1 or changed[i]:
            continue

        if not agrees_with_neighors(value, i, fixed_exps):
            left, curr, right = fixed_exps[i-1:i+2]
            fixed_exps[i] = infer_genre_exp(value, fixed_levels[i], left, right)
            changed[i] = True

    # For unchanged values
    # fix disagreement with star level
    for i in range(1, num_elements - 1):
        value = original_exps[i]
        if value == -1 or changed[i]:
            continue
        
        min_exp, max_exp = get_exp_range_from_level(fixed_levels[i])
        is_agree = min_exp <= value <= max_exp
        if not is_agree:
            curr = original_exps[i]
            left = find_non_negative(fixed_exps, i, 'left')
            right = find_non_negative(fixed_exps, i, 'right')
            fixed_exps[i] = infer_genre_exp(value, fixed_levels[i], left, right)
            changed[i] = True

    # Final pass
    # "Blanket" fix -1s to a conservative (lower) value.
    min_exp = min(fixed_exps)
    for i in range(1, num_elements - 1):
        if fixed_exps[i] == -1:
            curr = original_exps[i]
            left = find_non_negative(fixed_exps, i, 'left')
            right = find_non_negative(fixed_exps, i, 'right')
            fixed_exps[i] = infer_genre_exp(curr, fixed_levels[i], left, right)
            # min_exp_level, _ = get_exp_range_from_level(fixed_levels[i])
            # fixed_exps[i] = max(min_exp_level, min_exp)
            changed[i] = True

    return fixed_exps, changed


def find_non_negative(elements, index, direction='left'):
    current_index = index
    step = -1 if direction == 'left' else 1

    while 0 <= current_index < len(elements):
        if elements[current_index] <= -1:
            return elements[current_index]
        current_index += step

    if direction == 'left':
        return None
    else:
        return 0

def to_genre_exp_row(player):
    """Given the 8 traits, return the 12 genre levels (e.g. 0-40+)."""
    name2exp = {guess.name:exp for guess,exp in zip(player.genre_guesses, player.genre_exps)}

    exps = []
    for name in GENRES:
        exp = name2exp.get(name, 0)
        
        # TODO: should this -1 check be here?
        if exp == -1:
            exp = 0

        exps.append(exp)

    return exps


def to_genre_lvl_row(player):
    """Given the 8 traits, return the 12 genre levels (e.g. 0,1,2,3,4)."""
    name2level = {guess.name:lvl for guess,lvl in zip(player.genre_guesses, player.genre_levels)}

    levels = []
    for name in GENRES:
        level = name2level.get(name, 0)
        levels.append(level)
    
    return levels
