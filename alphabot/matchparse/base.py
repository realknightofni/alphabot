import os

import cv2
import imagehash
import numpy as np

from PIL import Image

DEFAULT_HASH_SIZE = 10

def draw_bboxes(image, bboxes, rgb_color, thickness=2):
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    for x, y, w, h in bboxes:
        cv2.rectangle(image, (x, y), (x+w, y+h), bgr_color, thickness=thickness)


def draw_contours(image, contours, rgb_color, thickness=1, draw_bbox=False):
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

    if draw_bboxes:
        bboxes = [cv2.boundingRect(c) for c in contours]
        draw_bboxes(image, bboxes, rgb_color, thickness)
    else:
        cv2.drawContours(image, contours, -1, bgr_color, thickness)


def draw_text(
    img,
    *,
    text,
    uv_top_left,
    color=(255, 255, 255),
    fontScale=0.5,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0, 0, 0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline.
    """
    text = str(text)

    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]


def add_text_top_left(
    image,
    text,
    font_scale=1.0,
    thickness=2,
    position=(10, 120),
    rgb_color=(0, 244, 207),
    outline_color=(0, 0, 0),
    ):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    color = (rgb_color[2], rgb_color[1], rgb_color[0])

    draw_text(
        image,
        text=text,
        uv_top_left=position,
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        outline_color=outline_color,
    )
