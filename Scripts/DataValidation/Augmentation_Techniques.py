import numpy as np
from scipy.interpolate import CubicSpline

def jitter(data, sigma=0.03):
    """Applies jittering by adding Gaussian noise."""
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise

def scaling(data, sigma=0.2):
    """Applies scaling by multiplying by a random factor."""
    factor = np.random.normal(1, sigma)
    return data * factor

def magnitude_warping(data, sigma=0.2, num_knots=4):
    """Applies magnitude warping using a cubic spline."""
    orig_shape = data.shape
    warped = np.zeros_like(data)
    for i in range(data.shape[1]):
        # Generate random knots
        knots = np.random.normal(1, sigma, num_knots)
        knot_x = np.linspace(0, data.shape[0], num_knots)
        spline = CubicSpline(knot_x, knots)
        warp = spline(np.arange(data.shape[0]))
        warped[:, i] = data[:, i] * warp
    return warped

def slicing(data, window_size=50):
    """Applies slicing by taking a random window of the data."""
    start = np.random.randint(0, data.shape[0] - window_size)
    return data[start:start + window_size]

def permutation(data, segment_length=100):
    """Applies permutation by shuffling segments of the data."""
    segments = np.split(data, np.arange(segment_length, data.shape[0], segment_length))
    np.random.shuffle(segments)
    return np.concatenate(segments)

def time_warping(data, sigma=0.2, num_knots=4):
    """Applies time warping using a cubic spline."""
    orig_shape = data.shape
    warped = np.zeros_like(data)
    for i in range(data.shape[1]):
        # Generate random knots
        knots = np.random.normal(1, sigma, num_knots)
        knot_x = np.linspace(0, data.shape[0], num_knots)
        spline = CubicSpline(knot_x, knots)
        warp_path = spline(np.arange(data.shape[0]))
        # Apply time warping
        for j in range(data.shape[0]):
            new_index = int(j * warp_path[j])
            if new_index < data.shape[0]:
                warped[j, i] = data[new_index, i]
            else:
                warped[j, i] = data[-1, i]
    return warped
