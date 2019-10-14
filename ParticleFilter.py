import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as  plt

class ParticleFilter:
    def __init__(self, num_particles):
        self.number_particles = num_particles
        self.threshold = 0.463
        self.actual_crack = np.random.normal(loc=0.01, scale=5e-4, size=self.number_particles)
        self.m = np.random.normal(loc=4.0, scale=0.2, size=self.number_particles)
        self.c = np.random.lognormal(mean=-22.33, sigma=1.12, size=self.number_particles)

    def predict(self):
        predicted_crack = ParticleFilter.getCrack(self.actual_crack, self.m, self.c)
        self.actual_crack = predicted_crack

    def reSample(self, measured):
        weights = ParticleFilter.likelihood(measured, self.actual_crack)
        self.weights = weights / np.sum(weights)                                                                  #normalizing the weights
        temp_m = np.random.choice(self.m, size=self.number_particles, replace=True, p=self.weights.reshape(-1))
        temp_c = np.random.choice(self.c, size=self.number_particles, replace=True, p=self.weights.reshape(-1))
        # now we resample from the posterior density using the above sampled weights
        actual_predicted_crack = np.random.choice(self.actual_crack, size=self.number_particles, replace=True, p=self.weights.reshape(-1))
        self.actual_crack = actual_predicted_crack
        self.m, self.c = temp_m, temp_c

    @staticmethod
    def getCrack(previous_crack, m, C):
        stress_range = 78
        dN = 50
        stress_range_old_crack = stress_range * np.sqrt(np.pi * previous_crack)
        left = C * np.power(stress_range_old_crack , m) * dN
        return (left + previous_crack)


    @staticmethod
    def likelihood(measured_crack, actual_crack):
        sigma = np.sqrt(np.log(1 + np.power(0.001 / actual_crack, 2)))
        mu = np.log(actual_crack) - 1 / 2 * np.power(sigma, 2)
        dist = lognorm([sigma], loc=mu)
        likeli = dist.pdf(measured_crack)
        return likeli.reshape(-1)

    def get_prediction(self):
        return self.actual_crack


if __name__ == "__main__":
    measured_crack = [0.0119, 0.0103, 0.0118, 0.0085, 0.0122,0.0110,0.0120,0.0113,0.0122]
    plt.figure()
    plt.plot(measured_crack,"o")
    plt.show()
    part_obj = ParticleFilter(10000)
    for meas in measured_crack:
        part_obj.predict()
        part_obj.reSample(meas)
        predictions = part_obj.get_prediction()

        plt.hist(predictions)
        plt.show()
