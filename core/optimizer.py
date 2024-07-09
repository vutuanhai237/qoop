import numpy as np
from ..backend import constant
from ..core import gradient


def sgd(thetas: np.ndarray, grad_loss: np.ndarray) -> np.ndarray:
    """Standard gradient descent

    Args:
        - thetas (np.ndarray): parameters
        - grad_loss (np.ndarray): gradient value

    Returns:
        - np.ndarray: new params
    """
    thetas -= constant.LEARNING_RATE * grad_loss
    return thetas


def adam(
    thetas: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    iteration: int,
    grad_loss: np.ndarray,
) -> np.ndarray:
    """Adam Optimizer. Below codes are copied from somewhere :)

    Args:
        - thetas (np.ndarray): parameters
        - m (np.ndarray): params for Adam
        - v (np.ndarray): params for Adam
        - i (int): params for Adam
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    num_thetas = thetas.shape[0]
    beta1, beta2, epsilon = constant.BETA1, constant.BETA2, constant.EPSILON
    for i in range(0, num_thetas):
        m[i] = beta1 * m[i] + (1 - beta1) * grad_loss[i]
        v[i] = beta2 * v[i] + (1 - beta2) * grad_loss[i] ** 2
        mhat = m[i] / (1 - beta1 ** (iteration + 1))
        vhat = v[i] / (1 - beta2 ** (iteration + 1))
        thetas[i] -= constant.LEARNING_RATE * mhat / (np.sqrt(vhat) + epsilon)
    return thetas


def qng_fubini_study_hessian(
    thetas: np.ndarray, G: np.ndarray, grad_loss: np.ndarray
) -> np.ndarray:
    """Type of QNG

    Args:
        - thetas (np.ndarray): parameters
        - G (np.ndarray): Fubini-study matrix
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    thetas = np.real(thetas - constant.LEARNING_RATE * (np.linalg.pinv(G) @ grad_loss))
    return thetas


def qng_fubini_study(
    thetas: np.ndarray, G: np.ndarray, grad_loss: np.ndarray
) -> np.ndarray:
    """Type of QNG

    Args:
        - thetas (np.ndarray): parameters
        - G (np.ndarray): Fubini-study matrix
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    thetas = np.real(thetas - constant.LEARNING_RATE * (np.linalg.pinv(G) @ grad_loss))
    return thetas


def qng_fubini_study_scheduler(
    thetas: np.ndarray, G: np.ndarray, iter: int, grad_loss: np.ndarray
) -> np.ndarray:
    """Type of QNG

    Args:
        - thetas (np.ndarray): parameters
        - G (np.ndarray): Fubini-study matrix
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    lr = constant.LEARNING_RATE * constant.GAMMA ** round(iter / 30)
    thetas = np.real(thetas - lr * ((np.linalg.pinv(G) @ grad_loss)))
    return thetas


def qng_qfim(
    thetas: np.ndarray, psi: np.ndarray, grad_psi: np.ndarray, grad_loss: np.ndarray
) -> np.ndarray:
    """Update parameters based on quantum natural gradient algorithm
    \n thetas^{i + 1} = thetas^{i} - alpha * F^{-1} * nabla L

    Args:
        - thetas (np.ndarray): parameters
        - psi (np.ndarray): current state
        - grad_psi (np.ndarray): all partial derivatives of $\psi$, is a N x N matrix
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    F = gradient.qfim(psi, grad_psi)
    # Because det(QFIM) can be nearly equal zero
    if np.isclose(np.linalg.det(F), 0):
        inverse_F = np.identity(F.shape[0])
    else:
        inverse_F = np.linalg.pinv(F)
    thetas -= constant.LEARNING_RATE * (inverse_F @ grad_loss)
    return thetas


def qng_adam(
    thetas: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    i: int,
    psi: np.ndarray,
    grad_psi: np.ndarray,
    grad_loss: np.ndarray,
) -> np.ndarray:
    """After calculating the QFIM, use it in Adam optimizer

    Args:
        - thetas (np.ndarray): parameters
        - m (np.ndarray): params for Adam
        - v (np.ndarray): params for Adam
        - i (int): params for Adam
        - psi (np.ndarray): current state
        - grad_psi (np.ndarray): all partial derivatives of $\psi$, is a N x N matrix
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    F = gradient.qfim(psi, grad_psi)
    # Because det(QFIM) can be nearly equal zero
    if np.isclose(np.linalg.det(F), 0):
        inverse_F = np.identity(F.shape[0])
    else:
        inverse_F = np.linalg.pinv(F)

    grad = inverse_F @ grad_loss
    thetas = adam(thetas, m, v, i, grad)
    return thetas
