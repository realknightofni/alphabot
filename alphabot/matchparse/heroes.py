import cv2


# TODO: these should not be flat pixel numbers, it should scale w/ something (e.g. size of a reference object, or entire image)
MIN_WIDTH = 30
MIN_HEIGHT = 30

EXPECTED_HEROES = 8

# TODO: consolidate into a qa.py file or something
def highlight_and_save_contours(image, contours, filename):
    image_copy = image.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 1)
    cv2.imwrite(filename, image_copy)



def get_hero_bboxes(image, max_x=None):
    if max_x is None:
        _, width = image.shape[:2]
        max_x = width//2
    left_side = image[:, :max_x]

    gray = cv2.cvtColor(left_side, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite('testheroes.png', binary)

    image_copy = image.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # cv2.imwrite('testheroes_contours.png', image_copy)

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

        # print(f'{cx},{cy},{cw},{ch} - {aspect_ratio} {is_squareish} - {len(approx)}')

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            bbox = (x, y, w, h)
            bboxes.append(bbox)

    inferred_bboxes = []
    if len(bboxes) < EXPECTED_HEROES:
        expected_width = max(b[2] for b in bboxes)
        expected_height = max(b[3] for b in bboxes)
        expected_size = max(expected_width, expected_height)

        for approx, contour in zip(polygons, square_contours):
            x, y, w, h = cv2.boundingRect(approx)
            
            if w <= expected_size:
                bbox = x, y, w, h
                inferred_bboxes.append(bbox)

    # highlight_and_save_contours(image, square_contours, 'testheroes_squares.png')

    if inferred_bboxes:
        hero_bboxes = inferred_bboxes
    else:
        hero_bboxes = bboxes

    if len(hero_bboxes) > EXPECTED_HEROES:
        print(f'WARNING: more than {EXPECTED_HEROES} hero bounding boxes detected {len(bboxes)} + {len(inferred_bboxes)}')  # TODO: make this an actual warning
    # print(len(hero_bboxes))
    # print(sorted(hero_bboxes, key=lambda x: x[1]))
    return hero_bboxes
