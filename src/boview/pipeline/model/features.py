"""Functions for extracting features from a height frame and mask"""

import math

import numpy as np


def matlab_round(x: float, decimals: int = 0) -> float:
    """
    Rounds a value to the nearest decimal, with ties away from zero

    Parameters:
    - x: value to round
    - decimals: number of decimal places

    Returns:
    - rounded: rounded value
    """
    multiplier = 10**decimals

    return math.copysign(math.floor(abs(x) * multiplier + 0.5), x) / multiplier


def get_length(mask: np.ndarray) -> np.integer:
    """
    Finds the greatest number of foreground pixels in a row

    Parameters:
    - mask: HxW mask of an animal rotated to horizontal

    Returns:
    - length: maximum row occupancy in pixels
    """
    return np.count_nonzero(mask, axis=1).max()


def get_width(mask: np.ndarray) -> np.integer:
    """
    Finds the greatest number of foreground pixels in a column

    Parameters:
    - mask: HxW mask of an animal rotated to horizontal

    Returns:
    - width: maximum column occupancy in pixels
    """
    return np.count_nonzero(mask, axis=0).max()


def get_dorsal_area(mask: np.ndarray) -> int:
    """
    Counts foreground pixels in the dorsal silhouette

    Parameters:
    - mask: HxW segmentation mask

    Returns:
    - dorsal_area: area in square pixels
    """
    return np.count_nonzero(mask)


def get_volume(height_frame: np.ndarray) -> np.float64:
    """
    Sums masked heights after converting millimeters to centimeters

    Parameters:
    - height_frame: HxW height frame in millimeters, zero outside the mask

    Returns:
    - volume: summed height in square pixels times centimeters
    """
    return height_frame.sum(dtype=np.float64) / 10.0


def get_max_height(height_frame: np.ndarray) -> np.floating:
    """
    Finds the maximum masked height

    Parameters:
    - height_frame: HxW height frame in millimeters, zero outside the mask

    Returns:
    - max_height: maximum height in centimeters
    """
    return height_frame.max() / 10.0


def get_avg_height(volume: float, dorsal_area: int) -> float:
    """
    Calculates mean height over the dorsal silhouette

    Parameters:
    - volume: summed height in square pixels times centimeters
    - dorsal_area: positive foreground area in square pixels

    Returns:
    - avg_height: average height in centimeters
    """
    return volume / dorsal_area


def get_z(fl: float, avg_height: float) -> float:
    """
    Calculates camera distance to the average animal height

    Parameters:
    - fl: camera height above the floor in millimeters
    - avg_height: average animal height in centimeters

    Returns:
    - z: distance from camera to animal in millimeters
    """
    return fl - avg_height * 10.0


def get_centroid(mask: np.ndarray) -> tuple[int, int]:
    """
    Finds the rounded centroid of the foreground pixels

    Parameters:
    - mask: HxW segmentation mask

    Returns:
    - centroid: (row, column) pixel coordinates
    """
    rows, cols = np.nonzero(mask)

    return int(np.round(rows.mean())), int(np.round(cols.mean()))


def get_centroid_height(
    height_frame: np.ndarray, centroid: tuple[int, int], radius: int = 50
) -> np.floating:
    """
    Finds the maximum height in a window around the centroid

    Parameters:
    - height_frame: HxW height frame in millimeters, zero outside the mask
    - centroid: (row, column) pixel coordinates inside the frame
    - radius: positive half-width of the window in pixels

    Returns:
    - centroid_height: maximum window height in centimeters
    """
    height = height_frame
    row, col = centroid
    roi = height[
        max(0, row - radius) : min(height.shape[0], row + radius),
        max(0, col - radius) : min(height.shape[1], col + radius),
    ]

    return roi.max() / 10.0


def get_circumference_min_tot(height_frame: np.ndarray, col: int, fl: float) -> float:
    """
    Sums rounded vertical segments exceeding the flat pixel length

    Parameters:
    - height_frame: HxW height frame in millimeters, zero outside the mask
    - col: column index to measure
    - fl: camera height above the floor in millimeters

    Returns:
    - circumference: calibrated contour length, or -1 for fewer than two heights
    """
    height = height_frame.astype(np.float64, copy=False)
    values = height[:, col]
    indices = np.flatnonzero(values > 0)
    if indices.size < 2:
        return -1.0

    total = 0.0
    for row in range(indices[0], indices[-1]):
        d0, d1 = values[row : row + 2]
        if d0 <= 0 or d1 <= 0:
            continue

        distance = fl / 10.0 - (d0 + d1) / 20.0
        if distance <= 0:
            continue

        pixel_length = matlab_round(1.0 / (316.51 * distance**-0.862), 1)
        segment = matlab_round(math.hypot(pixel_length, (d1 - d0) / 10.0), 1)
        if segment > pixel_length:
            total += segment

    return total


def get_comp_max_tot_vertical_max_column(
    height_frame: np.ndarray, mask: np.ndarray, fl: float
) -> tuple[float, np.integer | None]:
    """
    Measures circumference at the column containing the most valid heights

    Parameters:
    - height_frame: HxW height frame in millimeters
    - mask: HxW segmentation mask aligned with the height frame
    - fl: camera height above the floor in millimeters

    Returns:
    - circumference: calibrated contour length, or -1 if unavailable
    - col: first widest column index, or None if fewer than two valid heights
    """
    height = height_frame

    valid = mask.astype(bool) & (height > 0)
    counts = np.count_nonzero(valid, axis=0)
    col = np.argmax(counts)
    if counts[col] < 2:
        return -1.0, None

    return get_circumference_min_tot(np.where(valid, height, 0), col, fl), col


def get_surface_area_matlab_exact(height_frame: np.ndarray, fl: float) -> np.float64:
    """
    Sums rounded triangle areas exceeding the notebook's flat-area threshold

    Parameters:
    - height_frame: HxW height frame in millimeters, zero outside the mask
    - fl: camera height above the floor in millimeters

    Returns:
    - surface_area: calibrated area using the notebook's triangle formula and rounding
    """
    height = height_frame.astype(np.float64, copy=False)
    origin = height[:-1, :-1]
    right = height[:-1, 1:]
    bottom = height[1:, :-1]
    diagonal = height[1:, 1:]

    distances = [
        fl / 10.0 - (other + origin) / 20.0 for other in (right, bottom, diagonal)
    ]
    dx, dy, dxy = [
        np.round(1.0 / (316.51 * distance**-0.862), 1) for distance in distances
    ]
    dz_x, dz_y, dz_xy = [(other - origin) / 10.0 for other in (right, bottom, diagonal)]

    area_a = np.round(
        0.5
        * np.sqrt((dz_x * dxy) ** 2 + (dz_x * dxy - dx * dz_xy) ** 2 + (dx * dxy) ** 2),
        1,
    )
    area_b = np.round(
        0.5
        * np.sqrt((dy * dz_xy - dz_y * dxy) ** 2 + (dz_y * dxy) ** 2 + (dy * dxy) ** 2),
        1,
    )

    threshold = np.round(dx * dy / 2.0, 1)

    return area_a[area_a > threshold].sum() + area_b[area_b > threshold].sum()


def get_hip_heights(
    height_frame: np.ndarray, keypoints: np.ndarray
) -> tuple[np.floating, np.floating]:
    """
    Samples heights at hip keypoints 11 and 15

    Parameters:
    - height_frame: HxW height frame in millimeters, zero outside the mask
    - keypoints: (N, 2) pixel coordinates in (x, y) order, aligned with the frame

    Returns:
    - top_hip_height: height at keypoint 11 in centimeters
    - bottom_hip_height: height at keypoint 15 in centimeters
    """
    height = height_frame
    hips = np.round(keypoints[[10, 14], :2]).astype(int)

    return tuple(height[y, x] / 10.0 for x, y in hips)


def get_average_hip_height(
    top_hip_height: float, bottom_hip_height: float, avg_height: float
) -> float:
    """
    Averages hip heights that are positive and at least half the mean body height

    Parameters:
    - top_hip_height: top hip height in centimeters
    - bottom_hip_height: bottom hip height in centimeters
    - avg_height: average body height in centimeters

    Returns:
    - average_hip_height: mean valid hip height in centimeters, or -1 if neither is valid
    """
    valid = [
        height
        for height in (top_hip_height, bottom_hip_height)
        if height > 0 and height >= avg_height / 2.0
    ]

    return np.mean(valid) if valid else -1.0


def get_hip_width(height_frame: np.ndarray, keypoints: np.ndarray) -> np.integer | int:
    """
    Measures the positive-height span at the midpoint column of the hips

    Parameters:
    - height_frame: HxW height frame in millimeters, zero outside the mask
    - keypoints: (N, 2) pixel coordinates in (x, y) order, aligned with the frame

    Returns:
    - hip_width: distance between outermost positive rows in pixels, or -1 if unavailable
    """
    height = height_frame
    hips = np.round(keypoints[[10, 14], :2]).astype(int)
    col = hips[:, 0].sum() // 2
    indices = np.flatnonzero(height[:, col] > 0)

    return indices[-1] - indices[0] if indices.size >= 2 else -1


def get_features(
    height_frame: np.ndarray, mask: np.ndarray, keypoints: np.ndarray, fl: float
) -> dict:
    """
    Extracts the 20 model inputs from aligned, postprocessed inputs

    Parameters:
    - height_frame: HxW height frame in millimeters, rotated to horizontal
    - mask: HxW segmentation mask rotated with the frame
    - keypoints: (N, 2) (x, y) pixel coordinates rotated with the frame
    - fl: camera height above the floor in millimeters

    Returns:
    - features: 10 original measurements followed by their 10 transformed counterparts
      in model input order, using Z = FL - AvgHeight * 10 in millimeters;
      suffixed names preserve the notebook's transformations, including scaling
      heights already in centimeters; statistical normalization is not applied
    """
    mask = mask.astype(bool)
    height = height_frame
    height = np.where(mask, height, 0)

    dorsal_area = get_dorsal_area(mask)
    volume = get_volume(height)
    avg_height = get_avg_height(volume, dorsal_area)

    centroid = get_centroid(mask)
    circumference, col = get_comp_max_tot_vertical_max_column(height, mask, fl)
    top_hip, bottom_hip = get_hip_heights(height, keypoints)

    # Original measurements, including extras available for further analysis
    measurements = {
        "Length": get_length(mask),
        "Width": get_width(mask),
        "DorsalArea": dorsal_area,
        "Volume": volume,
        "CentroidHeight": get_centroid_height(height, centroid),
        "MaxHeight": get_max_height(height),
        "AvgHeight": avg_height,
        "Z": get_z(fl, avg_height),
        "SurfaceArea": get_surface_area_matlab_exact(height, fl),
        "CompMaxTot_VerticalMaxColumn": circumference,
        "CompMaxTot_ColumnIndex": col,
        "CircumferenceMinTot_CentroidColumn": get_circumference_min_tot(
            height, centroid[1], fl
        ),
        "HipWidth": get_hip_width(height, keypoints),
        "AverageHipHeight": get_average_hip_height(top_hip, bottom_hip, avg_height),
        "TopHipHeight": top_hip,
        "BottomHipHeight": bottom_hip,
        "FL": fl,
    }

    # Distance-based conversions using average height converted to millimeters
    z = measurements["Z"]
    length_factor = 0.143 * (z / 1000) ** 0.862
    area_factor = 0.023 * (z / 1000) ** 1.866

    # Model inputs in the same order as the modelling notebook
    features = {
        "Length": measurements["Length"],
        "Width": measurements["Width"],
        "DorsalArea": measurements["DorsalArea"],
        "Volume": measurements["Volume"],
        "CentroidHeight": measurements["CentroidHeight"],
        "MaxHeight": measurements["MaxHeight"],
        "AvgHeight": measurements["AvgHeight"],
        "SurfaceArea": measurements["SurfaceArea"],
        "HipWidth": measurements["HipWidth"],
        "AverageHipHeight": measurements["AverageHipHeight"],
        "Length_cm": measurements["Length"] * length_factor,
        "Width_cm": measurements["Width"] * length_factor,
        "DorsalArea_cm2": measurements["DorsalArea"] * area_factor,
        "Volume_cm3": measurements["Volume"] * area_factor,
        "CentroidHeight_cm": measurements["CentroidHeight"] * length_factor,
        "MaxHeight_cm": measurements["MaxHeight"] * length_factor,
        "AvgHeight_cm": measurements["AvgHeight"] * length_factor,
        "SurfaceArea_cm2": measurements["SurfaceArea"] * area_factor,
        "HipWidth_cm": measurements["HipWidth"] * length_factor,
        "AverageHipHeight_cm": measurements["AverageHipHeight"] * length_factor,
    }

    return features
