"""Logic to determine placement order.

General Algorithm:
 - Find the "PLAYER" header
 - find all text underneath that start under the "PLAYER" header
 - Assign placements based off the vertical spacing of each text


"""
import numpy as np

UNKNOWN_PLAYER_NAME = 'UNKNOWN_PLAYER'


def get_minmax(bounding_box):
    xs = []
    ys = []
    for x, y in bounding_box:
        xs.append(x)
        ys.append(y)

    return min(xs), max(xs), min(ys), max(ys)


def get_center(bounding_box):
    minmax = get_minmax(bounding_box)

    xcen = (minmax[0] + minmax[1]) / 2
    ycen = (minmax[2] + minmax[3]) / 2

    return float(xcen), float(ycen)


def get_player_header(easyocr_results):
    px0 = None
    for i, (bbox, text, confidence) in enumerate(easyocr_results):
        if text in {'PLAYER', '玩家'}:
            px0, px1, py0, py1 = get_minmax(bbox)
            break

    if px0 is None:
        raise ValueError('Could not find the "PLAYER" banner at the top')

    return i, px0, px1, py0, py1


def get_players(easyocr_results):
    """Get the list of player names from the easyocr results.

    Parameters
    ----------
    easyocr_results : list
        produced from reader.readtext from easyocr

    Returns
    -------
    list of tuples
        (player_name, ycenter, confidence, result_index)

    """
    pi, px0, px1, py0, py1 = get_player_header(easyocr_results)
    
    players = []
    for i, (bbox, text, confidence) in enumerate(easyocr_results):
        if i < pi+1:
            continue

        startx = bbox[0][0]
        if startx >= px0 and startx <= px1:
            xcen, ycen = get_center(bbox)
            player = (text, ycen, confidence, i, bbox)
            players.append(player)

    players = sorted(players, key=lambda p: p[1])  # sort by ycenter
    return merge_nearby_players(players)


def merge_nearby_players(players, y_threshold=10):
    if not players:
        return []
    
    merged_players = []
    current_player = list(players[0])  # Convert to list for mutability
    
    for next_player in players[1:]:
        current_y = current_player[1]
        next_y = next_player[1]
        
        if abs(next_y - current_y) < y_threshold:
            # Merge with current player
            current_player[0] += " " + next_player[0]  # Concatenate text
            current_player[1] = (current_y + next_y) / 2  # Average y position
            current_player[2] = (current_player[2] + next_player[2]) / 2  # Average confidence
            # Update bbox to be the union of both
            current_player[4] = merge_bboxes(current_player[4], next_player[4])
        else:
            # Add current player to results and start new
            merged_players.append(tuple(current_player))
            current_player = list(next_player)
    
    # Add the last player
    merged_players.append(tuple(current_player))
    return merged_players


def merge_bboxes(bbox1, bbox2):
    all_points = bbox1 + bbox2
    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]
    
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    return (
        (min_x, min_y),  # top-left
        (max_x, min_y),  # top-right
        (max_x, max_y),  # bottom-right
        (min_x, max_y)   # bottom-left
    )


def get_player_placements(easyocr_results):
    """Determine placements based on the vertical placement of each player name."""
    pi, px0, px1, py0, py1 = get_player_header(easyocr_results)  # TODO: rewrite this eventually, this gets run twice
    players = get_players(easyocr_results)

    if len(players) > 8:
        # Notes: there may be some ways to get around this? e.g. remove low confidence score text
        # needs more consideration
        raise ValueError(f'More than 8 names found in the final score screen. Cannot parse player placements.: {players}')

    ydeltas = [players[i+1][1] - players[i][1] for i in range(len(players)-1)]
    median_spacing = np.median(ydeltas)
    expected_1st_spacing = median_spacing * 0.73  # spacing between the bottom of "PLAYER" header and 1st place

    placement = 1
    placements = []
    previous_y = py1
    i = 0
    while len(placements) < 8:
        expected_spacing = expected_1st_spacing if placement == 1 else median_spacing

        # TODO: there is a neater way to organize this code
        if i > len(players)-1:
            expected_ycen = previous_y + expected_spacing
            placements.append((placement, UNKNOWN_PLAYER_NAME, 0, expected_ycen, None))
            previous_y = expected_ycen
            placement += 1
            continue

        player_name, ycen, confidence, _, bbox = players[i]

        y_delta = ycen - previous_y
        # print(f'spacing: {median_spacing} / {expected_spacing}  / {y_delta} >>> {abs(y_delta - expected_spacing)} {0.15 * expected_spacing}')
        # print(f'{player_name} | {ycen} - {previous_y} = {y_delta} | {expected_spacing}')
        if abs(y_delta - expected_spacing) / expected_spacing <= 0.18:  # within % of expected spacing
            placements.append((placement, player_name, confidence, ycen, bbox))
            previous_y = ycen
            i += 1
        else:
            print('Could not find player within expected spacing - adding UNKNOWN_PLAYER')
            expected_ycen = previous_y + expected_spacing
            placements.append((placement, UNKNOWN_PLAYER_NAME, 0, expected_ycen, None))
            previous_y = expected_ycen

        placement += 1

    return placements


def get_reporter_placement(placements, image):
    y_centers = [p[3] for p in placements]
    bboxes = [p[4] for p in placements]

    min_x = min([b[0][0] for b in bboxes if b])

    square_size = 2
    pad_size = 2
    x = int(min_x - square_size - pad_size)

    colors = []
    brightnesses = []
    for y_center in y_centers:
        y = int(y_center - (square_size // 2))

        roi = image[y:y+square_size, x:x+square_size]
        pixels = roi.reshape(-1, 3)
        color = np.median(pixels, axis=0).astype(int)
        colors.append(color)
        
        brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]  # luminance equation
        brightnesses.append(brightness)
    
    brightest_index = np.argmax(brightnesses)

    placement = placements[brightest_index]
    print(f'Reporter: {placement}')

    return placement[0], placement[1]   # TODO: this should probably be an actual object

    






