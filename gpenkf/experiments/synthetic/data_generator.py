import numpy as np

from gpenkf.experiments.data_provider import DataProvider


class DataGenerator(DataProvider):

    def __init__(self, borders, sample_size, f, noise, validation_size):
        self.borders = borders
        self.sample_size = sample_size
        self.f = f
        self.noise = noise
        self.x_validation, self.f_validation = self.generate_sample_of_size(validation_size)

    def generate_sample(self):
        return self.generate_sample_of_size(self.sample_size)

    def generate_sample_of_size(self, input_sample_size):
        x_new = (self.borders[1] - self.borders[0]) * np.random.random_sample((input_sample_size, 1)) + self.borders[0]
        x_new = np.sort(x_new, axis=0)
        f_new = self.f(np.squeeze(x_new))
        f_new_noised = f_new + np.random.normal(loc=0., scale=self.noise, size=(input_sample_size,))

        return x_new, f_new_noised