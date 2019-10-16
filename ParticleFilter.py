import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as  plt

"""Author: Spandan Mishra. Particle filter based remaining useful life estimaton"""
class ParticleFilter:
    def __init__(self, num_particles):
        self.number_particles = num_particles
        self.threshold = 0.015
        self.actual_crack = np.random.normal(loc=0.01, scale=5e-4, size=self.number_particles)
        self.m = np.random.normal(loc=4.0, scale=0.2, size=self.number_particles)
        self.c = np.random.normal(loc=-22.33, scale=1.12, size=self.number_particles)
        self.sigma = 0.001


    def predict(self):
        predicted_crack = ParticleFilter.getCrack(self.actual_crack, self.m, self.c)
        self.actual_crack = predicted_crack

    def update(self, weights, measured):
        weights *= ParticleFilter.likelihood(measured, self.actual_crack, self.sigma)
        weights += 1.e-300 # avoid round off to zero
        weights /= np.sum(weights)                                     # normalizing the weights

    def resampleFromIndex(self, weights):
        temp_m = ParticleFilter.simpleResample(particles=self.m, weights=weights)
        temp_c = ParticleFilter.simpleResample(particles=self.c, weights=weights)
        actual_predicted_crack = ParticleFilter.simpleResample(particles=self.actual_crack, weights=weights)
        self.actual_crack = actual_predicted_crack
        self.m, self.c = temp_m, temp_c

    @staticmethod
    def neff(weights):
        """if no new measurements. do not resample
        if Neff fall below some threshold it is time to resample.
         Useful starting point is N/2. We could also use N"""
        return 1./ np.sum(np.square(weights))

    @staticmethod
    def getCrack(previous_crack, m, C):
        stress_range = 78
        dN = 50
        stress_range_old_crack = stress_range * np.sqrt(np.pi * previous_crack)
        left = np.exp(C) * np.power(stress_range_old_crack, m) * dN
        return left + previous_crack

    @staticmethod
    def likelihood(measured_crack, actual_crack, std_dev):
        sigma = np.sqrt(np.log(1 + np.power(std_dev / actual_crack, 2)))
        mu = np.log(actual_crack) - 0.5 * np.power(sigma, 2)
        return lognorm.pdf(measured_crack, s=sigma, loc=mu,scale=1).reshape(-1)

    def get_prediction(self):
        return self.actual_crack

    def get_posterior_pred(self):
        """Posterior predictive model"""
        self.predict()
        previous_crack = self.get_prediction()
        stdev = self.sigma
        sigma = np.sqrt(np.log(1 + np.power(stdev / previous_crack, 2)))
        mu = np.log(previous_crack) - 0.5 * np.power(sigma, 2)
        posterior_crack_dist = np.random.lognormal(mean=mu, sigma=sigma, size=self.number_particles).reshape(-1)
        self.actual_crack = posterior_crack_dist


    @staticmethod
    def simpleResample(particles, weights):
        N = len(particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1
        indexes = np.searchsorted(cumulative_sum, np.random.rand(N))
        # Resample according to index
        particles[:] = particles[indexes]
        weights.fill(1.0 / N)
        return particles

    def estimate(self, weight):
        mean = np.mean(self.actual_crack)
        var = np.var(self.actual_crack)
        return mean, var

    @property
    def prognosis(self):
        """This functions gives the probablity of failure of structure at any given point"""
        all_predictions = self.get_prediction()
        values_exceeding_threshold = all_predictions > self.threshold
        prob_failure = np.sum(values_exceeding_threshold) / len(all_predictions)
        return prob_failure


class RemainingUsefulLife:
    def __init__(self, all_predictions, total_time, percentile, threshold):
        self.predictions = all_predictions
        self.total_time = total_time
        self.percentile = percentile
        self.threshold = threshold
        self.RUL =[]

    def getRUL(self, t):
        N = len(self.total_time)
        for i in range(self.predictions.shape[1]):
            loc = np.argmax(self.predictions[:, i] > self.threshold)
            if loc == 0:   # case when simulation does not exceed threshold
                temp = self.total_time[N-1] - self.total_time[t-1]
            else:#case when it exceeds threshold
                temp = self.total_time[loc-1] - self.total_time[t-1]
            self.RUL.append(temp)
        return self.RUL




def main(number_particles=5000, measured_crack=[]):
    measured_crack = measured_crack
    all_predictions = []
    number_particles = number_particles
    part_obj = ParticleFilter(number_particles)
    measured_data_iterator = 0
    time_array = []
    init_time = 0
    average_crack = []
    average_variance = []
    prob_failure = []
    weights = np.full(shape=number_particles, fill_value=1. / number_particles)
    mu, var = 0.0, 0.0
    while mu < part_obj.threshold:
        if measured_data_iterator < len(measured_crack):
            for meas in measured_crack:
                # use the priors from the previous step to make prediction
                part_obj.predict()
                # update the priors using likelikehood and we obtain posterior
                part_obj.update(weights=weights, measured=meas)
                if part_obj.neff(weights) < number_particles / 2:
                    part_obj.resampleFromIndex(weights)
                    assert np.allclose(weights, 1 / number_particles)
                all_predictions.append(part_obj.get_prediction())
                mu, var = part_obj.estimate(weights)
                if mu >= part_obj.threshold:
                    break
                average_crack.append(mu)
                average_variance.append(var)
                init_time += 50
                time_array.append(init_time)
                measured_data_iterator += 1
                prob_failure.append(part_obj.prognosis)

        else:
            # this is the prediction step we do not need to resample because
            # there are no measurements to re-smaple and update the posterior
            part_obj.predict()
            part_obj.resampleFromIndex(weights)
            mu, var = part_obj.estimate(weights)
            all_predictions.append(part_obj.get_prediction())
            if mu >= part_obj.threshold:
                break
            average_crack.append(mu)
            average_variance.append(var)
            init_time += 50
            time_array.append(init_time)
            prob_failure.append(part_obj.prognosis)
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

    # Fragility
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
