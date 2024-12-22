import numpy as np

class AMSGradOptimizer:
    def __init__(self, stepsize =0.001, beta1=0.8, beta2=0.999, epsilon=1e-8):
        """
        Initialize the AMSGrad optimizer.

        Args:
            stepsize (float): Learning rate.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small constant for numerical stability.
        """
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.v_hat = None
        self.t = 0

    def step(self, theta, grad_fn):
        """
        Perform a single optimization step using AMSGrad.

        Args:
            grad_fn (callable): Function to compute the gradient of the objective function.
            theta (numpy.ndarray): Current parameters.

        Returns:
            numpy.ndarray: Updated parameters.
        """
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
            self.v_hat = np.zeros_like(theta)
        grad = grad_fn(theta)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        v_1 = self.v
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(grad, 2)
        v_hat = np.maximum(v_1, self.v)

        theta = theta - self.stepsize * self.m / (np.sqrt(v_hat) + self.epsilon)
        return theta

    def step_and_cost(self, objective_fn, theta, grad_fn):
        """
        Perform a single optimization step and return the updated parameters and cost.

        Args:
            objective_fn (callable): Function to compute the cost of the objective function.
            grad_fn (callable): Function to compute the gradient of the objective function.
            theta (numpy.ndarray): Current parameters.

        Returns:
            tuple: Updated parameters and the cost of the objective function.
        """
        theta = self.step(theta, grad_fn)
        cost = objective_fn(theta)
        return theta, cost

class NadamOptimizer:
    def __init__(self, stepsize =0.001, beta1=0.8, beta2=0.999, epsilon=1e-8):
        """
        Initialize the Nadam optimizer.

        Args:
            stepsize (float): Learning rate.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small constant for numerical stability.
        """
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, theta, grad_fn):
        """
        Perform a single optimization step using Nadam.

        Args:
            grad_fn (callable): Function to compute the gradient of the objective function.
            theta (numpy.ndarray): Current parameters.

        Returns:
            numpy.ndarray: Updated parameters.
        """
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        grad = grad_fn(theta)
        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(grad, 2)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        theta = theta -  self.stepsize / (np.sqrt(v_hat) + self.epsilon) * (self.beta1 * m_hat + (1 - self.beta1)
                                                                           * grad / (1 - self.beta1**self.t))

        return theta

    def step_and_cost(self, objective_fn, theta, grad_fn):
        """
        Perform a single optimization step and return the updated parameters and cost.

        Args:
            objective_fn (callable): Function to compute the cost of the objective function.
            grad_fn (callable): Function to compute the gradient of the objective function.
            theta (numpy.ndarray): Current parameters.

        Returns:
            tuple: Updated parameters and the cost of the objective function.
        """
        theta = self.step(theta, grad_fn)
        cost = objective_fn(theta)
        return theta, cost

class AdamaxOptimizer:
    def __init__(self, stepsize =0.001, beta1=0.8, beta2=0.999, epsilon=1e-8):
        """
        Initialize the Nadam optimizer.

        Args:
            stepsize (float): Learning rate.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small constant for numerical stability.
        """
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, theta, grad_fn):
        """
        Perform a single optimization step using Nadam.

        Args:
            grad_fn (callable): Function to compute the gradient of the objective function.
            theta (numpy.ndarray): Current parameters.

        Returns:
            numpy.ndarray: Updated parameters.
        """
        self.t += 1
        if self.m is None:
            self.m = np.zeros(np.shape(theta))
            self.v = np.zeros(np.shape(theta))
        grad = grad_fn(theta)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = np.maximum(self.beta2 * self.v, np.abs(grad))

        m_hat = self.m / (1 - self.beta1**self.t)

        theta = theta - self.stepsize * m_hat / (self.v + self.epsilon)
        return theta

    def step_and_cost(self, objective_fn, theta, grad_fn):
        """
        Perform a single optimization step and return the updated parameters and cost.

        Args:
            objective_fn (callable): Function to compute the cost of the objective function.
            grad_fn (callable): Function to compute the gradient of the objective function.
            theta (numpy.ndarray): Current parameters.

        Returns:
            tuple: Updated parameters and the cost of the objective function.
        """
        theta = self.step(theta, grad_fn)
        cost = objective_fn(theta)
        return theta, cost
    
class QHAdamOptimizer:
    def __init__(self, stepsize =0.001, beta1=0.8, beta2=0.999, epsilon=1e-8, v1: float = 0.7, v2: float = 1.0):
        """
        Initialize the Nadam optimizer.

        Args:
            stepsize (float): Learning rate.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small constant for numerical stability.
        """
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v1 = v1
        self.v2 = v2
        self.m = None
        self.v = None
        self.t = 0

    def step(self, theta, grad_fn):
        """
        Perform a single optimization step using Nadam.

        Args:
            grad_fn (callable): Function to compute the gradient of the objective function.
            theta (numpy.ndarray): Current parameters.

        Returns:
            numpy.ndarray: Updated parameters.
        """
        self.t += 1
        if self.m is None:
            self.m = np.zeros(np.shape(theta))
            self.v = np.zeros(np.shape(theta))
        grad = grad_fn(theta)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(grad, 2)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        theta = theta - self.stepsize * ((1 - self.v1) * grad + self.v1 * m_hat) / (np.sqrt((1 - self.v2) * np.power(grad, 2) + self.v2 * v_hat) + self.epsilon)

        return theta

    def step_and_cost(self, objective_fn, theta, grad_fn):
        """
        Perform a single optimization step and return the updated parameters and cost.

        Args:
            objective_fn (callable): Function to compute the cost of the objective function.
            grad_fn (callable): Function to compute the gradient of the objective function.
            theta (numpy.ndarray): Current parameters.

        Returns:
            tuple: Updated parameters and the cost of the objective function.
        """
        theta = self.step(theta, grad_fn)
        cost = objective_fn(theta)
        return theta, cost
    
class RAdamOptimizer:
    def __init__(self, stepsize =0.001, beta1=0.8, beta2=0.999):
        """
        Initialize the Nadam optimizer.

        Args:
            stepsize (float): Learning rate.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small constant for numerical stability.
        """
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.p_max = 2 / (1 - self.beta2) - 1
        self.t = 0
        self.m = None 
        self.v = None

    def step(self, theta, grad_fn):
        """
        Perform a single optimization step using Nadam.

        Args:
            grad_fn (callable): Function to compute the gradient of the objective function.
            theta (numpy.ndarray): Current parameters.

        Returns:
            numpy.ndarray: Updated parameters.
        """
        self.t += 1
        if self.m is None:
            self.m = np.zeros(np.shape(theta))
            self.v = np.zeros(np.shape(theta))
        grad = grad_fn(theta) + 10**(-20)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = 1 / self.beta2 * self.v + (1 - self.beta2) * np.power(grad, 2)
        m_hat = self.m / (1 - self.beta1**self.t)
        p_t = self.p_max - 2 * self.t * self.beta2**self.t / (1 - self.beta2**self.t)

        if p_t > 4:
            l_t = np.sqrt((1 - self.beta2**self.t) / self.v)
            r_t = np.sqrt(((p_t - 4) * (p_t - 2) * self.p_max) / ((self.p_max - 4) * (self.p_max - 2) * p_t))
            theta = theta - self.stepsize * r_t * m_hat * l_t
        else:
            theta = theta - self.stepsize * m_hat
        return theta

    def step_and_cost(self, objective_fn, theta, grad_fn):
        """
        Perform a single optimization step and return the updated parameters and cost.

        Args:
            objective_fn (callable): Function to compute the cost of the objective function.
            grad_fn (callable): Function to compute the gradient of the objective function.
            theta (numpy.ndarray): Current parameters.

        Returns:
            tuple: Updated parameters and the cost of the objective function.
        """
        theta = self.step(theta, grad_fn)
        cost = objective_fn(theta)
        return theta, cost
    
class AdamWOptimizer:
    def __init__(self, stepsize =0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7, weight_decay: float = 0.01):
        """
        Initialize the Nadam optimizer.

        Args:
            stepsize (float): Learning rate.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small constant for numerical stability.
        """
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.m = None 
        self.v = None

    def step(self, theta, grad_fn):
        """
        Perform a single optimization step using Nadam.

        Args:
            grad_fn (callable): Function to compute the gradient of the objective function.
            theta (numpy.ndarray): Current parameters.

        Returns:
            numpy.ndarray: Updated parameters.
        """
        self.t += 1
        if self.m is None:
            self.m = np.zeros(np.shape(theta))
            self.v = np.zeros(np.shape(theta))
        grad = grad_fn(theta)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(grad, 2)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        theta = theta - self.stepsize * m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * grad
        return theta

    def step_and_cost(self, objective_fn, theta, grad_fn):
        """
        Perform a single optimization step and return the updated parameters and cost.

        Args:
            objective_fn (callable): Function to compute the cost of the objective function.
            grad_fn (callable): Function to compute the gradient of the objective function.
            theta (numpy.ndarray): Current parameters.

        Returns:
            tuple: Updated parameters and the cost of the objective function.
        """
        theta = self.step(theta, grad_fn)
        cost = objective_fn(theta)
        return theta, cost