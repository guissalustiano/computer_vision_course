# Refenrences:
# - https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html
# - https://stackoverflow.com/questions/45817325/opencv-python-cv2-perspectivetransform
# - https://answers.opencv.org/question/144252/perspective-transform-without-crop/

from dataclasses import dataclass
from math import ceil
from itertools import combinations
from random import shuffle

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


@dataclass
class Point:
    x: int
    y: int
    id: str = None
    color: str = 'black'

    @staticmethod
    def from_array(arr, id=None, color=None):
        return Point(arr[0], arr[1], id, color)

    def as_array(self):
        return np.array([self.x, self.y])

    def as_tuple(self):
        return (self.x, self.y)

    def as_shapely(self):
        return SPoint(self.x, self.y)

    def distance(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

    def __iter__(self):
        return iter(self.as_tuple())

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __div__(self, other):
        return Point(self.x / other, self.y / other)

    @property
    def hash(self):
        return self.id or self.color or self.as_tuple()


@dataclass
class Frame:
    top_left: Point
    top_right: Point
    bottom_left: Point
    bottom_right: Point

    @property
    def top(self):
        return min(self.top_left.y, self.top_right.y)

    @property
    def bottom(self):
        return max(self.bottom_left.y, self.bottom_right.y)

    @property
    def left(self):
        return min(self.top_left.x, self.bottom_left.x)

    @property
    def right(self):
        return max(self.top_right.x, self.bottom_right.x)

    @property
    def width(self):
        return np.abs(self.right - self.left)

    @property
    def height(self):
        return np.abs(self.bottom - self.top)

    @staticmethod
    def from_image(img):
        return Frame(
            top_left = Point(0, 0),
            top_right = Point(img.shape[1], 0),
            bottom_left = Point(0, img.shape[0]),
            bottom_right = Point(img.shape[1], img.shape[0])
        )

    def __iter__(self):
        return iter((self.top_right, self.top_left, self.bottom_left, self.bottom_right))

def unzip(l: list[tuple]):
    """
    Example:
    >>> unzip([(1, 2), (3, 4)])
    [(1, 3), (2, 4)]
    """
    return tuple(list(zip(*l)))

def getTransform(psx: list[Point], psy: list[Point]):
    """
    Calculate the transform matrix to transform from two list of four points
    """

    # Cast list[Point] to np.array
    psx = np.float32([p.as_array() for p in psx])
    psy = np.float32([p.as_array() for p in psy])

    # Calculate transform matrix
    return cv.getPerspectiveTransform(psx, psy)

def perspectiveTransformPoint(M, point: Point):
    """
    Aplly perspective transform to a point
    """
    return Point.from_array(
        cv.perspectiveTransform(np.float32([[point.as_array()]]), M)[0][0],
        point.id,
        point.color,
    )

def warpPerspectiveFrame(M, frame):
    """
    Aplly perspective transform to every point in a frame.

    The usage with Frame.from_image(img) is similar to matlab "outputLimits" method.
    """

    return Frame(
        top_left = perspectiveTransformPoint(M, frame.top_left),
        top_right = perspectiveTransformPoint(M, frame.top_right),
        bottom_left = perspectiveTransformPoint(M, frame.bottom_left),
        bottom_right = perspectiveTransformPoint(M, frame.bottom_right)
    )


def warpPerspectiveImage(M, img, frame=None):
    """
    Aplly perspective transform to a image.
    Automatically calculate the size of the output image.
    Don't handle with negative transformation output.
    """
    frame = frame or warpPerspectiveFrame(M, Frame.from_image(img))
    size = (int(frame.right), int(frame.bottom))
    img_p = cv.warpPerspective(img, M, size)
    return img_p

def error(psx, psy):
    """
    Calculate the average error between same points in two set of points
    """
    sum = 0
    qtd = 0
    for px in psx:
        for py in psy:
            if px.hash == py.hash:
                sum += px.distance(py)
                qtd += 1
    return sum / qtd

def points_by_hash(ps):
    """
    Return a dict with points grouped by hash (id or color)
    """
    return {p.hash: p for p in ps}

def zip_same_hash(psx, psy):
    """
    Group points by hash
    """
    d_psx = points_by_hash(psx)
    d_psy = points_by_hash(psy)
    keys = set(d_psx.keys()) & set(d_psy.keys())

    return [
        (d_psx[k], d_psy[k])
        for k in keys
    ]


def select_points(raw_psx, raw_psy):
    """
    Brute force method to select the best four points to transform
    """
    ps = zip_same_hash(raw_psx, raw_psy)

    best_comb = None
    best_error = 10e100

    combs = combinations(ps, 4)
    for comb in combs:
        psx, psy = unzip(comb)
        M = getTransform(psy, psx)
        new_psy = [perspectiveTransformPoint(M, p) for p in raw_psy]
        err = error(raw_psx, new_psy)
        if err < best_error:
            best_comb = comb
            best_error = err

    psx, psy = unzip(best_comb)

    assert len(psx) == len(psy) == 4

    return psx, psy

def transform(imgx, raw_psx, imgy, raw_psy):
    """
    Stitched two images based on set of points with at least 4 points in common
    """

    psx, psy = select_points(raw_psx, raw_psy)

    # This transformation can generate negative values, so we need to shift it
    My = getTransform(psy, psx)

    # Calculate frame after transformation
    fy = warpPerspectiveFrame(My, Frame.from_image(imgy))

    # How much we need to shift?
    offset = Point(max(0, -fy.left), max(0, -fy.top))

    # Apply offset to X points
    offseted_psx = [p + offset for p in psx]

    # Get transformation from shift points, now don't generate negative values
    My = getTransform(psy, offseted_psx)

    # X also need to be shifted to correct match with Y
    Mx = getTransform(psx, offseted_psx)

    # Apply transformation
    imgx = warpPerspectiveImage(Mx, imgx)
    imgy = warpPerspectiveImage(My, imgy)

    per_psx = [perspectiveTransformPoint(Mx, p) for p in raw_psx]
    per_psy = [perspectiveTransformPoint(My, p) for p in raw_psy]

    # Plot
    if True:
        fig, axs = plt.subplots(1, 1)
        draw_img(axs, imgx)

        fig, axs = plt.subplots(1, 1)
        draw_img(axs, imgy)

    # Join images
    img = np.zeros((
        max(imgx.shape[0], imgy.shape[0]),
        max(imgx.shape[1], imgy.shape[1]),
        3
    ), dtype=np.uint8)

    # Start with image Y
    img[:imgy.shape[0], :imgy.shape[1]] = imgy

    # Overlap image X ignoring black pixels
    mask = (imgx != 0).all(axis=2)
    img[:imgx.shape[0], :imgx.shape[1]][mask] = imgx[mask]

    return img, per_psx + per_psy

def transform_frame(img, current_frame, desire_frame, ps = []):
    """
    Apply transformation based on two frames
    """
    M = getTransform(
        [current_frame.top_left, current_frame.top_right, current_frame.bottom_left, current_frame.bottom_right],
        [desire_frame.top_left, desire_frame.top_right, desire_frame.bottom_left, desire_frame.bottom_right],
    )

    img = warpPerspectiveImage(M, img, frame=desire_frame)
    ps = [perspectiveTransformPoint(M, p) for p in ps]
    return img, ps

def draw_point(ax, point: Point):
    """
    Draw a point in a matplotlib axis
    """
    ax.plot(point.x, point.y, 'o', markersize=5, color=point.color)

def draw_img(ax, img, points: list[Point] = []):
    """
    Draw a image in a matplotlib axis
    """
    ax.imshow(img)

    for point in points:
        if point is None:
            continue
        draw_point(ax, point)

    ax.set(xticks=[], yticks=[])

def parse_points(pps):
    """
    Create a 4 points list with points of same lane with same color
    """
    colors = list(matplotlib.colors.CSS4_COLORS.keys())
    pss = unzip(pps)

    shuffle(colors)

    return tuple(
        [Point(p.x, p.y, id=p.id, color=c) for p, c in zip(ps, colors) if p is not None]
        for ps in pss
    )

def get_point_by_id(points, id):
    """
    Get a point by id in list of points
    """
    for point in points:
        if point.id == id:
            return point
    return None

def main():
    # Load images
    img1 = (cv.imread('West_wadden1.jpg'))
    img2 = (cv.imread('West_wadden2.jpg'))
    img3 = (cv.imread('West_wadden3.jpg'))
    img4 = (cv.imread('West_wadden4.jpg'))


    # related points
    ps1, ps2, ps3, ps4 = parse_points([
        (Point( 42,  372), Point(633,  357), None, None),
        (None, Point(154,  1159), Point(193, 173), None),
        (None, None, Point(961, 1135), Point(257, 1240)),
        (Point(114, 1578), Point(624, 1605), Point(690, 619), Point(56, 753)),
        (Point(534, 1609), Point(1055, 1622), Point(1113, 631), Point(459, 754)),
        (Point(618, 983), Point(1178, 982), Point(1122, 39), Point(526, 169)),
        (Point(39, 1103), Point(579, 1106), Point(604, 135), Point(26, 316)),
        # Corners
        (
            Point(1119, 122, "top_right_corner"),
            Point(71, 82, "top_left_corner"),
            Point(120, 1436, "bottom_left_corner"),
            Point(1001, 1445, "bottom_right_corner"),
        ),
        # Square
        (Point(913, 1086, "top_left_square"), None, None, None),
        (Point(968, 1181, "bottom_left_square"), None, None, None),
    ])

    fig, axs = plt.subplots(2, 2)
    draw_img(axs[0][1], img1, ps1)
    draw_img(axs[0][0], img2, ps2)
    draw_img(axs[1][0], img3, ps3)
    draw_img(axs[1][1], img4, ps4)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)


    # Join images
    img, ps = transform(img1, ps1, img2, ps2)
    img, ps = transform(img, ps, img3, ps3)
    img, ps = transform(img, ps, img4, ps4)
    cv.imwrite("img.png", img)

    fig, axs = plt.subplots(1, 1)
    draw_img(axs, img) #, ps)

    # Cut borders
    desire_frame = Frame(
        top_left = Point(0, 0),
        top_right = Point(1800, 0),
        bottom_left = Point(0, 2200),
        bottom_right = Point(1800, 2200)
    )

    current_frame = Frame(
        top_left = get_point_by_id(ps, "top_left_corner"),
        top_right = get_point_by_id(ps, "top_right_corner"),
        bottom_left = get_point_by_id(ps, "bottom_left_corner"),
        bottom_right = get_point_by_id(ps, "bottom_right_corner")
    )

    img, ps = transform_frame(img, current_frame, desire_frame, ps)

    fig, axs = plt.subplots(1, 1)
    cv.imwrite("retificated.png", img)
    draw_img(axs, img)#, ps)

    return plt.show()




if __name__ == '__main__':
    main()
