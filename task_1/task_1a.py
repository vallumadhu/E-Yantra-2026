import cv2
import string
import argparse
import numpy as np


def get_aruco_markers(img) -> dict:

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    corners, ids, rejected = detector.detectMarkers(img)

    if ids is None or len(ids) != 4:
        raise ValueError(f"Expected 4 markers, got {0 if ids is None else len(ids)}")

    ids = ids.flatten()
    return dict(zip(ids,corners))



def get_inner_points(id2corners: dict) -> list:

    all_pts = np.vstack([c.reshape(4, 2) for c in id2corners.values()])
    centroid = all_pts.mean(axis=0)

    inner_points = {}

    for id,corners in id2corners.items():
        pts = corners.reshape(4, 2)
        dists = np.linalg.norm(pts - centroid, axis=1)
        inner_points[id] = pts[np.argmin(dists)]

    ordered_ids = [80, 85, 90, 95]
    src_points = []

    for id in ordered_ids:
        src_points.append(inner_points[id])

    return src_points



def perspective_transform(img, width: int, height: int):

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    marker_map = get_aruco_markers(gray)
    src_points = np.float32(get_inner_points(marker_map))
    dest_points = np.array([[0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1],
                            [0, height - 1],],dtype=np.float32)

    transformation_matrix = cv2.getPerspectiveTransform(src_points, dest_points)
    flat_img = cv2.warpPerspective(img, transformation_matrix, (width, height))

    return flat_img


def get_clean_mask(img,lower,upper):

    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv,lower,upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return cleaned


def get_centroids(binary_img) -> list:
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    MIN_CONTOUR_AREA = 200

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue

        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            continue

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        centroids.append((cx, cy))

    return centroids


def build_grid_labels() -> dict:

    NUM_LINES = 11
    CELL_SIZE = 75
    COLUMN_LABELS = string.ascii_uppercase[:NUM_LINES]

    grid_coords = np.arange(1, NUM_LINES + 1) * CELL_SIZE
    grid = {}

    for row in range(NUM_LINES):

        y = int(grid_coords[row])
        for col in range(NUM_LINES):

            x = int(grid_coords[col])
            label = f"{COLUMN_LABELS[col]}{row + 1}"
            grid[label] = (x, y)

    return grid


def match_to_grid(centroids: list,grid) -> list:

    closest = []

    labels = list(grid.keys())
    coords = np.array([grid[l] for l in labels])

    for cx, cy in centroids:

        dists = np.linalg.norm(coords - np.array([cx, cy]), axis=1)
        best_idx = np.argmin(dists)
        closest.append(labels[best_idx])

    return closest


def pipeline(IMAGE_PATH: str) -> None:

    img = cv2.imread(IMAGE_PATH)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    flat_img = perspective_transform(img,900,900)

    YELLOW_LOWER = np.array([20, 100, 100])
    YELLOW_UPPER = np.array([40, 255, 255])
    yellow_mask = get_clean_mask(flat_img,YELLOW_LOWER,YELLOW_UPPER)

    yellow_centroids = get_centroids(yellow_mask)

    grid = build_grid_labels()
    stable_survivors = match_to_grid(yellow_centroids,grid)

    RED_LOWER_1 = np.array([0, 100, 100])
    RED_UPPER_1 = np.array([10, 255, 255])

    RED_LOWER_2 = np.array([170, 100, 100])
    RED_UPPER_2 = np.array([180, 255, 255])

    red_mask1 = get_clean_mask(flat_img, RED_LOWER_1, RED_UPPER_1)
    red_mask2 = get_clean_mask(flat_img, RED_LOWER_2, RED_UPPER_2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    red_centroids = get_centroids(red_mask)
    critical_survivors = match_to_grid(red_centroids,grid)


    ids = list(get_aruco_markers(img).keys())


    lines = [
        f"Detected marker IDs: [{','.join(map(str, ids))}]",
        "",
        f"Critical Survivors: {', '.join(critical_survivors)}",
        f"Stable Survivors: {', '.join(stable_survivors)}",
    ]

    with open("results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)

    args = parser.parse_args()
    pipeline(args.image)