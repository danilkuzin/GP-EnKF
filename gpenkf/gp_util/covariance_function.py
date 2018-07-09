
class CovarianceFunction(object):
    def eval_cov_matrix(self, x1, x2):
        pass

    def predict(self, x_new, x_g, noise_variance, g):
        pass

    def log_likelihood(self, y, x_g, noise_variance):
        pass
