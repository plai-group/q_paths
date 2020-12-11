import numpy as np
from autograd import grad
import scipy.stats as st
from scipy import stats
import numpy as np
from copy import deepcopy
from tqdm.notebook import tqdm
from scipy.special import logsumexp
from autograd.scipy import stats as autograd_stats


class Student1D:
    def __init__(
        self, q=None, mean=None, variance=None, precision=None, precisionxmean=None, df = None
    ):
        self.df = (3-q)/(q-1) if (q is not None and q!=1) else df
        if self.df is None:
            raise ValueError("Please specify q parameter (q!=1) or df (degrees of freedom)")
        if mean is not None:
            self.mean = mean
            self.variance = variance
            self.precision = 1.0 / variance
            self.precisionxmean = mean / variance
        else:
            if precisionxmean is not None:
                self.precision = precision
                self.precisionxmean = precisionxmean
                self.variance = 1.0 / precision
                self.mean = precisionxmean / precision
            else:
                self.mean = 0
                self.variance = 1
                self.precision = 1.0 / variance
                self.precisionxmean = mean / variance
        self.sigma = np.sqrt(self.variance)
        # RV with functions: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
        #    e.g. logpdf(x, df, loc=0, scale=1),
        #         rvs(df, loc=0, scale=1, size=1, random_state=None) (sampling)
        self.student_t = stats.t
        self.student_t_autograd = autograd_stats.t
    def sample(self, no_samples):
        return self.student_t.rvs(self.df, size=no_samples, loc = self.mean, scale = self.sigma)
    def logprob(self, x):
        return self.student_t_autograd.logpdf(x, self.df, loc = self.mean, scale = self.sigma)


class Gaussian1D:
    def __init__(
        self, mean=None, variance=None, precision=None, precisionxmean=None
    ):
        if mean is not None:
            self.mean = mean
            self.variance = variance
            self.precision = 1.0 / variance
            self.precisionxmean = mean / variance
        else:
            self.precision = precision
            self.precisionxmean = precisionxmean
            self.variance = 1.0 / precision
            self.mean = precisionxmean / precision
        self.sigma = np.sqrt(self.variance)

    def sample(self, no_samples):
        return np.random.randn(no_samples) * np.sqrt(self.variance) + self.mean

    def logprob(self, x):
        return -(0.5 * (np.log(2 * np.pi * self.sigma * self.sigma) + ((x - self.mean) / self.sigma) ** 2))


def geometric_average(dist1, dist2, pow1=0.5, pow2=0.5):
    precision = dist1.precision * pow1 + dist2.precision * pow2
    precisionxmean = dist1.precisionxmean * pow1 + dist2.precisionxmean * pow2
    return Gaussian1D(precision=precision, precisionxmean=precisionxmean)


def moment_average(dist1, dist2, pow1=0.5, pow2=0.5):
    mean = pow1 * dist1.mean + pow2 * dist2.mean
    var = (
        pow1 * dist1.variance
        + pow2 * dist2.variance
        + pow1 * pow2 * (dist1.mean - dist2.mean) ** 2
    )
    var = np.abs(var)  # TODO: fix small negative variance
    return Gaussian1D(mean=mean, variance=var)


def run_alpha_average_loops(dist1, dist2, beta, alpha, no_iters=500):
    q = deepcopy(dist1)
    for i in range(no_iters):
        dist2hat = geometric_average(dist2, q, alpha, 1.0 - alpha)
        dist1hat = geometric_average(dist1, q, alpha, 1.0 - alpha)
        q = moment_average(dist1hat, dist2hat, 1 - beta, beta)
    return q


def find_average_batch(dist1, dist2, beta_list, option):
    dists = []
    for beta in beta_list:
        if option == "moment":
            dists.append(moment_average(dist1, dist2, 1.0 - beta, beta))
        elif option == "geometric":
            dists.append(geometric_average(dist1, dist2, 1.0 - beta, beta))
        else:
            raise NotImplementedError("unknown option")
    return dists


def find_alpha_average_batch(
    dist1, dist2, beta_list, alpha_list, no_iters=500
):
    aa_dists = []
    for alpha in alpha_list:
        aa_alpha = []
        for beta in beta_list:
            aa_alpha.append(
                run_alpha_average_loops(dist1, dist2, beta, alpha, no_iters)
            )
        aa_dists.append(aa_alpha)
    return aa_dists


def run_ais(beta_dists, no_samples):
    no_betas = len(beta_dists)
    dist0 = beta_dists[0]
    x = dist0.sample(no_samples)
    logw = np.zeros(no_samples)
    logwt = np.zeros(no_samples)
    for i in range(1, no_betas):
        di = beta_dists[i]
        dim1 = beta_dists[i - 1]
        # update lower bound weights
        logw = logw + di.logprob(x) - dim1.logprob(x)
        # perfect transition
        x = di.sample(no_samples)
        # update upper bound weights
        logwt = logwt + di.logprob(x) - dim1.logprob(x)

    logZ = logsumexp(logw) - np.log(no_samples)
    logZ_lower = np.mean(logw)
    logZ_upper = np.mean(logwt)
    return logZ, logZ_lower, logZ_upper


def find_ti_integrand_geometric(dists):
    mean1, var1 = dists[0].mean, dists[0].variance
    mean2, var2 = dists[-1].mean, dists[-1].variance
    mean_list = np.array([dist.mean for dist in dists])
    var_list = np.array([dist.variance for dist in dists])
    const_term = -0.5 * np.log(var2 / var1)
    f1 = 0.5 / var1 * ((mean_list - mean1) ** 2 + var_list)
    f2 = -0.5 / var2 * ((mean_list - mean2) ** 2 + var_list)
    f = f1 + f2 + const_term
    return f


def find_sum_elements_geometric(dists, beta_vec):
    f = find_ti_integrand_geometric(dists)
    elements_lower = f[:-1] * (beta_vec[1:] - beta_vec[:-1])
    elements_upper = f[1:] * (beta_vec[1:] - beta_vec[:-1])
    return elements_lower, elements_upper


def find_sum_elements_moment(dists, beta_vec):
    mean1, var1 = dists[0].mean, dists[0].variance
    mean2, var2 = dists[-1].mean, dists[-1].variance
    mean_list = np.array([dist.mean for dist in dists])
    var_list = np.array([dist.variance for dist in dists])
    logCb = (
        (1 - beta_vec) * -0.5 * np.log(2 * np.pi * var1)
        + beta_vec * -0.5 * np.log(2 * np.pi * var2)
        - 0.5 * mean1 ** 2 * (1 - beta_vec) / var1
        - 0.5 * mean2 ** 2 * beta_vec / var2
        + 0.5 * mean_list ** 2 / var_list
    )

    mk, mkm1 = mean_list[1:], mean_list[:-1]
    vk, vkm1 = var_list[1:], var_list[:-1]
    f1 = -0.5 / vk * ((mk - mkm1) ** 2 + vkm1 - vk)
    logCk, logCkm1 = logCb[1:], logCb[:-1]
    f2 = logCk - logCkm1
    elements_lower = f1 + f2

    f1 = 0.5 / vkm1 * ((mk - mkm1) ** 2 + vk - vkm1)
    logCk, logCkm1 = logCb[1:], logCb[:-1]
    f2 = logCk - logCkm1
    elements_upper = f1 + f2

    return elements_lower, elements_upper
