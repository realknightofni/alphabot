import cv2

import numpy as np

from .base import draw_bboxes

# TODO: these should not be flat pixel numbers, it should scale w/ something (e.g. size of a reference object, or entire image)
MIN_WIDTH = 20
MIN_HEIGHT = 20

EXPECTED_NUM_ARTIFACTS = 24


def get_artifact_bboxes(image, min_x=None):
    if min_x is None:
        _, width = image.shape[:2]
        min_x = width//4
    left_side = image[:, min_x:]

    gray = cv2.cvtColor(left_side, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite('testartifacts.png', binary)

    image_copy = image.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image_copy, (min_x+x, y), (min_x+x+w, y+h), (255, 0, 0), 2)
    # cv2.imwrite('testartifacts_contours.png', image_copy)

    bboxes = []
    square_contours = []
    polygons = []
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cw < MIN_WIDTH or ch < MIN_HEIGHT:
            continue

        aspect_ratio = float(cw) / ch  # NOTE: do you need the float here?
        is_squareish = 0.9 < aspect_ratio < 1.1
        if not is_squareish:
            continue

        square_contours.append(contour)

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygons.append(approx)

        # print(f'{min_x+cx},{cy},{cw},{ch} - {aspect_ratio} {is_squareish} - {len(approx)}')

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            bbox = (min_x+x, y, w, h)
            bboxes.append(bbox)

    inferred_bboxes = []
    if len(bboxes) < EXPECTED_NUM_ARTIFACTS:
        expected_min_width = min(b[2] for b in bboxes)
        expected_min_height = min(b[3] for b in bboxes)
        expected_min_size = min(expected_min_width, expected_min_height)

        expected_max_width = max(b[2] for b in bboxes)
        expected_max_height = max(b[3] for b in bboxes)
        expected_max_size = max(expected_max_width, expected_max_height)

        for approx, contour in zip(polygons, square_contours):
            x, y, w, h = cv2.boundingRect(approx)
            
            if expected_min_size <= w <= expected_max_size:
                bbox = min_x+x, y, w, h
                inferred_bboxes.append(bbox)

    # image_copy = image.copy()
    # draw_bboxes(image_copy, inferred_bboxes, (255, 255, 255), thickness=1)
    # cv2.imwrite('testartifacts_inferred.png', image_copy)

    if inferred_bboxes:
        artifact_bboxes = inferred_bboxes
    else:
        artifact_bboxes = bboxes

    median_width = np.median([b[2] for b in artifact_bboxes])
    artifact_bboxes = [b for b in artifact_bboxes if abs(b[2]-median_width) < 0.1 * median_width]

    if len(artifact_bboxes) > EXPECTED_NUM_ARTIFACTS:
        print(f'WARNING: more than {EXPECTED_NUM_ARTIFACTS} artifact bounding boxes detected {len(bboxes)} + {len(inferred_bboxes)}')  # TODO: make this an actual warning
    
    artifact_bboxes = [resize_artifact_box(b, median_width) for b in artifact_bboxes]
    # print(len(artifact_bboxes))
    # print(sorted(artifact_bboxes, key=lambda x: x[1]))
    return artifact_bboxes


def resize_artifact_box(artifact_bbox, forced_width=None):
    x, y, w, h = artifact_bbox

    # resize wide boxes - this sometimes occurs
    # for the third artifact - there's some white background
    # that the contour picks up
    if forced_width is not None:
        return x, y, int(forced_width), h
    elif w > h and (w-h)/w > 0.05:
        return x, y, h+1, h
    else:
        return x, y, w, h
