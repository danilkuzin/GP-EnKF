import numpy as np
import matplotlib.pyplot as plt


def plot_synthetic_function():
    borders = [-10, 10]  # The borders of the x-axis
    sample_size = 5  # Number of new observations at every iteration
    noise = 0.5  # Observation noise
    fine_grid = np.linspace(-10, 10, 2001)  # grid for plotting purposes

    def f(x):
        return x / 2 + (25 * x) / (1 + x ** 2) * np.cos(x)

    # Sample data example
    x_new = ((borders[1] - borders[0]) * np.random.random_sample((sample_size, 1))
             + borders[0])
    x_new = np.sort(x_new, axis=0)
    f_new = f(x_new)
    f_new_noised = f(x_new) + np.random.normal(loc=0., scale=noise,
                                               size=(sample_size, 1))

    # Plotting
    plt.plot(x_new, f_new, 'x', label='samples from f')
    plt.plot(x_new, f_new_noised, 'x', label='samples from f with noise')
    plt.plot(fine_grid, f(fine_grid), label='f')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)

    plt.savefig('synthetic_function_example.eps', format='eps')


if __name__ == "__main__":
    plot_synthetic_function()
