import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

"""Author: Spandan Mishra. Particle filter based remaining useful life estimation"""


class StateSpaceModel(ABC):
    @abstractmethod
    def sample_initial(self, num_particles: int) -> np.ndarray:
        """Return (N, d) array of initial particles."""

    @abstractmethod
    def transition(self, particles: np.ndarray) -> np.ndarray:
        """Propagate particles one step forward. Return (N, d)."""

    @abstractmethod
    def likelihood(self, measurement, particles: np.ndarray) -> np.ndarray:
        """Return (N,) likelihood of measurement given each particle."""

    def estimate(self, particles: np.ndarray, weights: np.ndarray):
        """Return (mean, var) — default uses weighted mean over all dims."""
        mean = np.average(particles, weights=weights, axis=0)
        var = np.average((particles - mean) ** 2, weights=weights, axis=0)
        return mean, var


class ParticleFilter:
    def __init__(self, num_particles: int, model: StateSpaceModel):
        self.num_particles = num_particles
        self.model = model
        self.particles = model.sample_initial(num_particles)   # shape (N, d)
        self.weights = np.full(num_particles, 1.0 / num_particles)

    def predict(self):
        self.particles = self.model.transition(self.particles)

    def update(self, measurement):
        self.weights *= self.model.likelihood(measurement, self.particles)
        self.weights += 1e-300  # avoid round-off to zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        N = self.num_particles
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        indexes = np.searchsorted(cumulative_sum, np.random.rand(N))
        self.particles = self.particles[indexes]  # fancy indexing → new array
        self.weights.fill(1.0 / N)

    def resample_if_needed(self):
        if self.neff() < self.num_particles / 2:
            self.resample()

    def neff(self):
        return 1.0 / np.sum(np.square(self.weights))

    def estimate(self):
        return self.model.estimate(self.particles, self.weights)

    def get_particles(self):
        return self.particles.copy()

    def prognosis(self, threshold):
        """Fraction of particles (crack size, column 0) exceeding threshold."""
        crack = self.particles[:, 0]
        return np.sum(crack > threshold) / self.num_particles


class CrackGrowthModel(StateSpaceModel):
    """Paris' Law fatigue crack growth model.

    State columns: [:, 0] = crack size, [:, 1] = m, [:, 2] = c
    """

    def __init__(self, sigma=0.001, stress_range=78, dN=50, threshold=0.015):
        self.sigma = sigma
        self.stress_range = stress_range
        self.dN = dN
        self.threshold = threshold

    def sample_initial(self, num_particles: int) -> np.ndarray:
        crack = np.random.normal(loc=0.01, scale=5e-4, size=num_particles)
        m = np.random.normal(loc=4.0, scale=0.2, size=num_particles)
        c = np.random.normal(loc=-22.33, scale=1.12, size=num_particles)
        return np.column_stack([crack, m, c])

    def transition(self, particles: np.ndarray) -> np.ndarray:
        crack = particles[:, 0]
        m = particles[:, 1]
        c = particles[:, 2]
        da = np.exp(c) * np.power(self.stress_range * np.sqrt(np.pi * crack), m) * self.dN
        particles[:, 0] = crack + da
        return particles

    def likelihood(self, measurement, particles: np.ndarray) -> np.ndarray:
        crack = particles[:, 0]
        sigma_ln = np.sqrt(np.log(1 + np.power(self.sigma / crack, 2)))
        mu_ln = np.log(crack) - 0.5 * np.power(sigma_ln, 2)
        return lognorm.pdf(measurement, s=sigma_ln, loc=mu_ln, scale=1).reshape(-1)

    def estimate(self, particles: np.ndarray, weights: np.ndarray):
        crack = particles[:, 0]
        return np.mean(crack), np.var(crack)


class RemainingUsefulLife:
    def __init__(self, all_predictions, total_time, percentile, threshold):
        self.predictions = all_predictions
        self.total_time = total_time
        self.percentile = percentile
        self.threshold = threshold
        self.RUL = []

    def getRUL(self, t):
        N = len(self.total_time)
        for i in range(self.predictions.shape[1]):
            loc = np.argmax(self.predictions[:, i] > self.threshold)
            if loc == 0:   # simulation does not exceed threshold
                temp = self.total_time[N - 1] - self.total_time[t - 1]
            else:          # exceeds threshold
                temp = self.total_time[loc - 1] - self.total_time[t - 1]
            self.RUL.append(temp)
        return self.RUL


def main(number_particles=5000, measured_crack=[]):
    all_predictions = []
    model = CrackGrowthModel()
    pf = ParticleFilter(number_particles, model)

    measured_data_iterator = 0
    time_array = []
    init_time = 0
    average_crack = []
    average_variance = []
    prob_failure = []
    mu, var = 0.0, 0.0

    while mu < model.threshold:
        if measured_data_iterator < len(measured_crack):
            for meas in measured_crack:
                pf.predict()
                pf.update(meas)
                pf.resample_if_needed()
                all_predictions.append(pf.get_particles()[:, 0])
                mu, var = pf.estimate()
                if mu >= model.threshold:
                    break
                average_crack.append(mu)
                average_variance.append(var)
                init_time += 50
                time_array.append(init_time)
                measured_data_iterator += 1
                prob_failure.append(pf.prognosis(model.threshold))
        else:
            pf.predict()
            pf.resample_if_needed()
            mu, var = pf.estimate()
            all_predictions.append(pf.get_particles()[:, 0])
            if mu >= model.threshold:
                break
            average_crack.append(mu)
            average_variance.append(var)
            init_time += 50
            time_array.append(init_time)
            prob_failure.append(pf.prognosis(model.threshold))
            measured_data_iterator += 1

    all_predictions = np.vstack(all_predictions)

    remain_obj = RemainingUsefulLife(all_predictions, time_array, [], 0.043)
    opt = remain_obj.getRUL(len(measured_crack))

    Ub = [x + 1.96 * np.sqrt(y) for x, y in zip(average_crack, average_variance)]
    Lb = [0.0 if x - 1.96 * np.sqrt(y) < 0 else x - 1.96 * np.sqrt(y) for x, y in zip(average_crack, average_variance)]

    plt.figure()
    plt.plot(time_array, average_crack, 'b')
    plt.plot(time_array, Lb, "g")
    plt.plot(time_array, Ub, "r")
    plt.xlabel("Cycles")
    plt.ylabel("Crack Size")
    plt.show()

    plt.figure()
    plt.plot(time_array, prob_failure, 'k')
    plt.xlabel("Cycles")
    plt.ylabel("Prob failure")
    plt.show()

    plt.figure()
    plt.hist(opt)
    plt.show()


if __name__ == "__main__":
    measured_crack = [0.0119, 0.0103]
    main(number_particles=100000, measured_crack=measured_crack)
