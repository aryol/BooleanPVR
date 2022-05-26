import itertools
import numpy as np


def create_test_matrix_11(rows, cols, p=0.5):
    """
    Returns a Boolean ({+1, -1}) random matrix of size rows * columns
    :param rows: number of rows
    :param cols: number of columns
    :param p: probability
    :return: a matrix with random bits of size rows * columns
    """
    return 1 - 2 * np.random.binomial(1, p, size=(rows, cols)).astype(np.float32)


def generate_all_binaries(d):
    """
    Generate all binary sequences of length d where bits are +1 and -1
    :param d: dimension
    :return: the output is a numpy array of size 2^d * d
    """
    return np.array([list(seq) for seq in itertools.product([-1, 1], repeat=d)], dtype=np.float32)


def calculate_Boolean_influence(function, dimension):
    """
    Calculates Boolean influences for different indices of the input function. Complexity of this function is O(2^d).
    :param function: input function
    :param dimension: the input dimension of the function
    :return: Numpy vector of Boolean influences for different indices of the input function
    """
    influences = []
    inputs = generate_all_binaries(dimension)
    outputs = function(inputs)
    for i in range(dimension):
        inputs_frozen = inputs.copy()
        inputs_frozen[:, i] = 1
        outputs_frozen = function(inputs_frozen)
        influences.append(np.mean((outputs - outputs_frozen) ** 2))
    return np.array(influences) / 2.0


def calc_PVR_function_truncated(X, pointer_size, window_size, function):
    """
    Calculates PVR on a given input matrix.
    Note that to convert bits to pointer, we assume -1, ..., -1 refers to the first window.
    :param X: input matrix (given in -1 and +1)
    :param pointer_size: Number of pointer bits.
    :param window_size: Size of window, i.e., number of bits in each window.
    :param function:  Aggregation function to be applied on the windows
    :return: The PVR result returned as a vector
    """
    powers_of_two = 2 ** np.arange(pointer_size - 1, -1, -1)
    return np.apply_along_axis(lambda row: function(row[int(powers_of_two @ (row[:pointer_size] + 1) / 2) + pointer_size: min(
        int(powers_of_two @ (row[:pointer_size] + 1) / 2) + window_size + pointer_size, X.shape[1])]), 1, X).ravel()



def calc_PVR_function_cyclic(X, pointer_size, window_size, function):
    """
    Calculates PVR on a given input matrix.
    Note that to convert bits to pointer, we assume -1, ..., -1 refers to the first window.
    :param X: input matrix (given in -1 and +1)
    :param pointer_size: Number of pointer bits.
    :param window_size: Size of window, i.e., number of bits in each window.
    :param function:  Aggregation function to be applied on the windows
    :return: The PVR result returned as a vector
    """
    powers_of_two = 2 ** np.arange(pointer_size - 1, -1, -1)
    X = np.hstack([X, X[:, pointer_size:]])
    return np.apply_along_axis(lambda row: function(row[int(powers_of_two @ (row[:pointer_size] + 1) / 2) + pointer_size:
        int(powers_of_two @ (row[:pointer_size] + 1) / 2) + window_size + pointer_size]), 1, X).ravel()


def calc_PVR_func_3bit(A, window, func):
    """
    Calculates PVR. Assumes pointer has three bits. Also assumes pointer bits are 0, 1 while other bits are +1, -1
    (as a more general function, see calc_PVR_function_truncated)
    :param A: Input
    :param window: Size of window. Window = 0 corresponds to having a single bit.
    :param func:  Function to be applied on the windows
    :return: The PVR result.
    """
    return np.apply_along_axis(lambda row: func(row[int(row[0] * 4 + row[1] * 2 + row[2]) + 3: min(
        int(row[0] * 4 + row[1] * 2 + row[2]) + window + 4, A.shape[1])]), 1, A).ravel()


def calculate_stair_case(X):
    """
    Calculates vanilla staircase function (x[0] + x[0]x[1] + x[0]x[1]x[2] + ... + x[0]...x[d-1]) on the input.
    This is f2 of the paper.
    :param X: Input samples as a matrix
    :return: Staircase function of the input samples
    """
    cumulative_products = np.zeros_like(X)
    for i in range(X.shape[1]):
        cumulative_products[:, i] = X[:, 0:i + 1].prod(axis=1)
    return cumulative_products.sum(axis=1).astype(np.float32)


def f1(X):
    """
    This is the f1 function mentioned in the paper (x[0]x[1] + ... + (d-1)x[d-2]x[d-1])
    :param X: Input samples as a matrix
    :return: f1 of the input samples
    """
    sqauare_terms = X[:, :-1] * X[:, 1:]
    sqauare_terms = sqauare_terms * (np.arange(X.shape[1] - 1) + 1)
    return sqauare_terms.sum(axis=1).astype(np.float32)


def f3(A, window=3):
    """
    This is the main PVR considered in the paper (3 pointer bits, w=3, and majority-vote aggregation)
    :param A: Input samples
    :param window: size of the window
    :return: f3 of the input samples
    """
    B = A.copy()
    B[:, 0:3] += 1
    B[:, 0:3] /= 2
    return calc_PVR_func_3bit(B, window - 1, lambda row: np.sign(row.sum()))


def calculate_fourier_coefficients(monomials, X, y):
    """
    calculate Fourier coefficients of monomials
    :param monomials: m * d, a mask to show which monomials we want. m is the number of monomials
    :param X: input data
    :param y: output data, y=f(X)
    :return: Fourier coefficient of the monomials which were indicated by monomials in the arguments.
    """
    return ((-2 * ((monomials @ ((1 - X.T) / 2)) % 2) + 1) @ y) / y.shape[0]


def create_monomials_mask(name_of_function, dimension, frozen_dim):
    """
    Creates a monomial mask for different functions
    :param name_of_function: name of the function
    :param dimension: dimension of the function
    :param frozen_dim: dimension to be frozen
    :return: suitable monomial mask based on function and frozen dimension
    """
    if name_of_function == 'staircase':
        mask = np.ones((dimension, dimension))
        mask = np.tril(mask)
        mask_frozen = mask.copy()
        mask_frozen[:, frozen_dim] = 0
        mask_frozen = mask_frozen[frozen_dim:]
        return np.vstack([mask, mask_frozen])
    elif name_of_function == 'f2':
        mask = np.eye(dimension)[:-1, :]
        mask = mask + np.roll(mask, 1, axis=1)
        if frozen_dim > 0:
            mask = np.vstack([mask, np.eye(dimension)[frozen_dim - 1, :]])
        if frozen_dim < dimension - 1:
            mask = np.vstack([mask, np.eye(dimension)[frozen_dim + 1, :]])
        return mask
    elif dimension <= 15:
        return (generate_all_binaries(dimension) + 1) / 2
    else:
        return None

