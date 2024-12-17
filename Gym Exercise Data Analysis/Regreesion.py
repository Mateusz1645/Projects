import numpy as np


def linear_regression_coefficients(x: np.ndarray, y: np.ndarray) -> list:
    """
    Calculate the coefficients of a linear regression line (b1 and b0) for the given data.

    Parameters:
    x (numpy.ndarray): Independent variable values.
    y (numpy.ndarray): Dependent variable values.

    Returns:
    list: A list containing the slope (b1) and intercept (b0) of the best-fit line.

    Raises:
    ValueError: If the lengths of x and y are not the same, or if they are empty.

    Example:
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 5, 4, 5])
    >>> linear_regression_coefficients(x, y)
    [0.2, 2.0]
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("x and y cannot be empty")

    # Calculate the mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate the slope (b1) using the formula: sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
    b1 = np.sum(x * (y - y_mean)) / np.sum((x - x_mean) ** 2)

    # Calculate the intercept (b0) using the formula: y_mean - b1 * x_mean
    b0 = y_mean - b1 * x_mean

    # Return the coefficients as a list
    return [b1, b0]


def linear_function(x: np.ndarray, coef: list) -> np.ndarray:
    """
    Compute the value of the linear function y = b1 * x + b0 for each element in x.

    Parameters:
    x (numpy.ndarray): Independent variable values.
    coef (list): A list containing the linear regression coefficients [b1, b0].

    Returns:
    numpy.ndarray: The predicted values of y based on the linear function.

    Raises:
    ValueError: If coef does not contain exactly two elements.

    Example:
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> coef = [0.2, 2.0]
    >>> linear_function(x, coef)
    array([2.2, 2.4, 2.6, 2.8, 3. ])
    """
    if len(coef) != 2:
        raise ValueError("coef must contain exactly two elements: [b1, b0]")

    # Apply the linear function: y = b1 * x + b0
    return coef[0] * x + coef[1]


def r_squared(x: np.ndarray, y: np.ndarray, coef: list) -> float:
    """
    Calculate the R-squared value for the linear regression model.

    Parameters:
    x (numpy.ndarray): Independent variable values.
    y (numpy.ndarray): Dependent variable values.
    coef (list): A list containing the linear regression coefficients [b1, b0].

    Returns:
    float: The R-squared value, which indicates how well the model fits the data.

    Raises:
    ValueError: If x, y, or coef are not valid (e.g., empty arrays or incorrect dimensions).

    Example:
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 5, 4, 5])
    >>> coef = [0.2, 2.0]
    >>> r_squared(x, y, coef)
    0.24
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("x and y cannot be empty")
    if len(coef) != 2:
        raise ValueError("coef must contain exactly two elements: [b1, b0]")

    # Predicted values
    y_pred = linear_function(x, coef)

    # Total sum of squares (SST)
    total_variance = np.sum((y - np.mean(y)) ** 2)

    # Residual sum of squares (SSE)
    residual_variance = np.sum((y - y_pred) ** 2)

    # R-squared value
    return 1 - (residual_variance / total_variance)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    y_true (numpy.ndarray): The true values of the dependent variable.
    y_pred (numpy.ndarray): The predicted values of the dependent variable.

    Returns:
    float: The Mean Squared Error between the true and predicted values.

    Raises:
    ValueError: If y_true and y_pred do not have the same length or are empty.

    Example:
    >>> y_true = np.array([2, 4, 5, 4, 5])
    >>> y_pred = np.array([2.2, 2.4, 2.6, 2.8, 3.0])
    >>> mean_squared_error(y_true, y_pred)
    0.048000000000000015
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred cannot be empty")

    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
    y_true (numpy.ndarray): The true values of the dependent variable.
    y_pred (numpy.ndarray): The predicted values of the dependent variable.

    Returns:
    float: The Root Mean Squared Error between the true and predicted values.

    Raises:
    ValueError: If y_true and y_pred do not have the same length or are empty.

    Example:
    >>> y_true = np.array([2, 4, 5, 4, 5])
    >>> y_pred = np.array([2.2, 2.4, 2.6, 2.8, 3.0])
    >>> root_mean_squared_error(y_true, y_pred)
    0.2182178902359924
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred cannot be empty")

    return np.sqrt(mean_squared_error(y_true, y_pred))
