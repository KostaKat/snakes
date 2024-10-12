import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates

def initialize_snake_contour(initial_snake_points, num_points=100):
    """
    Initialize the snake contour with interpolated points.
    """
    initial_snake_points = np.array(initial_snake_points, dtype=np.float64)
    # Interpolate to get num_points along the initial contour
    t = np.linspace(0, 1, len(initial_snake_points))
    t_interp = np.linspace(0, 1, num_points)
    x_interp = np.interp(t_interp, t, initial_snake_points[:, 1])
    y_interp = np.interp(t_interp, t, initial_snake_points[:, 0])
    snake = np.vstack([y_interp, x_interp]).T
    return snake

def calculate_edge_energy(image):
    """
    Compute the edge energy of the image.
    Edge energy is defined as the negative of the squared gradient magnitude.
    This creates minima at edges to attract the snake.
    """
    # Compute the gradient of the image
    grad_x = ndimage.sobel(image, axis=1, mode='reflect')
    grad_y = ndimage.sobel(image, axis=0, mode='reflect')
    # Compute the magnitude of the gradient
    grad_mag = np.hypot(grad_x, grad_y)
    # Normalize gradient magnitude to range [0,1]
    grad_mag = grad_mag / (grad_mag.max() + 1e-8)
    # Edge energy is negative of gradient magnitude squared
    edge_energy = - (grad_mag ** 2)
    return edge_energy

def calculate_line_energy(image):
    """
    Calculate the line energy based on image intensity.
    """
    # Normalize image intensity
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    # Line energy is negative of the intensity
    line_energy = -normalized_image
    return line_energy

def calculate_termination_energy(image):
    """
    Calculate the termination energy to detect corners and terminations.
    """
    # Compute second derivatives
    grad_xx = ndimage.sobel(image, axis=1, mode='reflect')
    grad_yy = ndimage.sobel(image, axis=0, mode='reflect')
    grad_xy = ndimage.sobel(grad_xx, axis=0, mode='reflect')

    # Compute the termination energy
    denominator = (1 + grad_xx ** 2 + grad_yy ** 2) ** 1.5
    termination_energy = (grad_yy * grad_xx - grad_xy ** 2) / denominator
    # Normalize
    termination_energy = termination_energy / (np.max(np.abs(termination_energy)) + 1e-8)
    return termination_energy

def calculate_scale_space_energy(image, sigma):
    """
    Calculate scale-space energy using Gaussian blur.
    """
    # Apply Gaussian blur to the image
    blurred_image = ndimage.gaussian_filter(image, sigma=sigma)
    # Use edge energy of blurred image
    scale_space_energy = calculate_edge_energy(blurred_image)
    return scale_space_energy

def compute_external_energy(image, w_line=0.0, w_edge=1.0, w_term=0.0, w_scale=0.0, sigma=2.0):
    """
    Compute the combined external energy.
    """
    # Individual energies
    line_energy = w_line * calculate_line_energy(image)
    edge_energy = w_edge * calculate_edge_energy(image)
    termination_energy = w_term * calculate_termination_energy(image)
    scale_space_energy = w_scale * calculate_scale_space_energy(image, sigma)

    # Combined external energy
    external_energy = line_energy + edge_energy + termination_energy + scale_space_energy
    return external_energy

def get_external_force(external_energy, snake):
    """
    Compute the external force (gradient of external energy) at the snake points.
    """
    # Compute the gradient of the external energy
    gy, gx = np.gradient(external_energy)
    # Interpolate the gradient at snake positions
    x = snake[:, 1]
    y = snake[:, 0]
    fx = map_coordinates(gx, [y, x], order=1, mode='reflect')
    fy = map_coordinates(gy, [y, x], order=1, mode='reflect')
    external_force = np.vstack([fy, fx]).T
    return external_force

def create_internal_matrix(n_points, alpha, beta, gamma):
    """
    Create the internal energy matrix A as per the original snake formulation.
    """
    # Coefficients
    a = beta
    b = -alpha - 4 * beta
    c = 2 * alpha + 6 * beta

    # Create the pentadiagonal matrix
    A = np.zeros((n_points, n_points))
    for i in range(n_points):
        A[i, i] = c
        A[i, (i - 1) % n_points] = b
        A[i, (i + 1) % n_points] = b
        A[i, (i - 2) % n_points] = a
        A[i, (i + 2) % n_points] = a

    # Create P = I + gamma * A
    P = np.eye(n_points) + gamma * A

    # Invert the matrix
    inv_P = np.linalg.inv(P)
    return inv_P

def evolve_snake(snake, inv_P, external_force, gamma):
    """
    Evolve the snake one iteration.
    """
    # Update snake positions
    snake = np.dot(inv_P, snake + gamma * external_force)
    return snake

def calculate_motion_energy(previous_snake, current_snake, w_motion):
    """
    Calculate motion energy to track snake across frames.
    """
    motion_energy = w_motion * (current_snake - previous_snake)
    return motion_energy

def calculate_stereo_energy(left_snake, right_snake, w_stereo):
    """
    Calculate stereo energy to match snakes between stereo image pairs.
    """
    stereo_energy = w_stereo * (left_snake - right_snake)
    return stereo_energy

def calculate_total_energy(snake, internal_energy_matrix, external_energy, motion_energy=0, stereo_energy=0):
    """
    Calculate the total energy of the snake.
    """
    # Internal energy term
    internal_energy = np.dot(snake.T, np.dot(internal_energy_matrix, snake))
    # External energy term
    external_energy_at_snake = map_coordinates(external_energy, [snake[:, 0], snake[:, 1]], order=1, mode='reflect')
    external_energy_term = np.sum(external_energy_at_snake)
    # Motion energy term
    total_motion_energy = np.sum(motion_energy)
    # Stereo energy term
    total_stereo_energy = np.sum(stereo_energy)
    # Total energy
    total_energy = internal_energy + external_energy_term + total_motion_energy + total_stereo_energy
    return total_energy