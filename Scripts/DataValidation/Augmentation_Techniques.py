import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import resample
from scipy.interpolate import interp1d

"""
Data Augmentation Techniques for Time Series Data

The following functions implement various data augmentation techniques for time series data. 
These techniques were inspired by the work of Iwana and Uchida (2021) in their paper "An empirical survey of data augmentation 
for time series classification with neural networks" (DOI: https://doi.org/10.1371/journal.pone.0254841 [11]. 
After conducting several tests, jittering and scaling emerged as the most effective methods, with jittering 
being selected as the primary augmentation technique in this project. Future iterations of this work 
may explore other methods or combinations for better results.

1. Jittering: Adds random noise to the data.
2. Scaling: Multiplies the data by a random factor.
3. Magnitude Warping: Warps the magnitude of the data by a smooth curve.
4. Slicing: Selects a random window of the time series.
5. Permutation: Shuffles segments of the time series.
6. Time Warping: Warps the time indices of the data.
7. Window Warping: Warps a specific portion of the time series.
"""

def jitter(data, sigma=0.03):
    """
    Applies jittering by adding Gaussian noise to the data.

    Parameters:
    - data (numpy array): The input time series data.
    - sigma (float): The standard deviation of the Gaussian noise.

    Returns:
    - numpy array: The augmented data with added noise.
    """
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise

def scaling(data, low=0.8, high=1.2):
    """
    Applies scaling by multiplying the data by a random factor within a specified range.

    Parameters:
    - data (numpy array): The input time series data.
    - low (float): The lower bound for the scaling factor.
    - high (float): The upper bound for the scaling factor.

    Returns:
    - numpy array: The scaled data.
    """
    factor = np.random.uniform(low, high)
    return data * factor

def magnitude_warping(data, sigma=0.2, num_knots=4):
    """
    Applies magnitude warping to the time series using a cubic spline.

    Parameters:
    - data (numpy array): The input time series data.
    - sigma (float): The standard deviation for the random knots.
    - num_knots (int): The number of knots used for the cubic spline.

    Returns:
    - numpy array: The warped data with adjusted magnitude.
    """
    orig_shape = data.shape
    warped = np.zeros_like(data)
    for i in range(data.shape[1]):
        knots = np.random.normal(1, sigma, num_knots)
        knot_x = np.linspace(0, data.shape[0], num_knots)
        spline = CubicSpline(knot_x, knots)
        warp = spline(np.arange(data.shape[0]))
        warped[:, i] = data[:, i] * warp
    return warped

def slicing(data, window_size=50):
    """
    Applies slicing by extracting a random window from the time series.

    Parameters:
    - data (numpy array): The input time series data.
    - window_size (int): The size of the window to extract.

    Returns:
    - numpy array: The sliced data window.
    """
    if data.shape[0] <= window_size:
        return data.copy()  # Return original data if it's too short
    start = np.random.randint(0, data.shape[0] - window_size)
    return data[start:start + window_size]

def permutation(data, segment_length=100):
    """
    Applies permutation by shuffling segments of the time series.

    Parameters:
    - data (numpy array): The input time series data.
    - segment_length (int): The length of each segment to shuffle.

    Returns:
    - numpy array: The permuted data.
    """
    segments = np.split(data, np.arange(segment_length, data.shape[0], segment_length))
    np.random.shuffle(segments)
    return np.concatenate(segments)

def time_warping(data, sigma=0.2, num_knots=4):
    """
    Applies time warping to the data by modifying the time indices with a smooth curve.

    Parameters:
    - data (numpy array): The input time series data.
    - sigma (float): The standard deviation for the random knots.
    - num_knots (int): The number of knots used for the cubic spline.

    Returns:
    - numpy array: The time-warped data.
    """
    orig_steps = np.arange(data.shape[0])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=num_knots)
    warp_positions = np.linspace(0, data.shape[0]-1, num_knots)
    warp_curve = CubicSpline(warp_positions, random_warps)(orig_steps)

    cumulative_warp = np.cumsum(warp_curve)
    cumulative_warp = (cumulative_warp / cumulative_warp[-1]) * (data.shape[0] - 1)

    warped = np.zeros_like(data)
    for i in range(data.shape[1]):
        interp_func = interp1d(cumulative_warp, data[:, i], kind='linear', fill_value="extrapolate")
        warped[:, i] = interp_func(orig_steps)
    return warped

def window_warp(data, window_ratio=0.05, scales=[0.98, 1.02]):
    """
    Applies window warping to a section of the time series.

    Parameters:
    - data (numpy array): The input time series data.
    - window_ratio (float): The size of the window to warp as a fraction of the total length.
    - scales (list): A range of scaling factors to apply to the window.

    Returns:
    - numpy array: The window-warped data.
    """
    length = data.shape[0]
    window_length = int(length * window_ratio)
    if window_length < 1:
        return data.copy()

    start = np.random.randint(0, length - window_length)
    end = start + window_length

    scale = np.random.uniform(scales[0], scales[1])

    window = data[start:end]
    resampled_window = resample(window, int(window_length * scale))

    warped = np.concatenate((data[:start], resampled_window, data[end:]), axis=0)
    warped = resample(warped, length)

    return warped
