import numpy as np

class QuantumCompilation():
    import qiskit, typing, types
    def __init__(self) -> None:
        self.u = None
        self.vdagger = None
        self.is_trained = False
        self.opt = None
        self.thetas = None
        self.thetass = []
        self.metrics = {}
        self.metrics_func = {}
        self.kwargs = None
        self.num_steps = 0
        return

    def __init__(self, u: types.FunctionType, opt, 
                 metrics_func: list[types.FunctionType], 
                 cost_func: types.FunctionType, thetas: np.ndarray = np.array([]), **kwargs):
        self.u = u
        self.is_trained = False
        self.opt = opt
        self.thetas = thetas
        self.thetass = []
        self.metrics = {}
        self.metrics_func = metrics_func
        self.cost_func = cost_func(u)
        self.kwargs = kwargs
        self.num_steps = 0
        self.costs = []
        return
    
    def fit(self, num_steps: int):
        for _ in range(num_steps):
            self.thetass.append(self.thetas)
            self.thetas, prev_cost = self.opt.step_and_cost(self.cost_func, self.thetas)
            self.costs.append(prev_cost)
        for metric_func in self.metrics_func:
            self.metrics[metric_func.__name__] = metric_func(self.thetas, self.u, self.vdagger, **self.kwargs)
        return 
    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.costs)
        plt.show()
        return