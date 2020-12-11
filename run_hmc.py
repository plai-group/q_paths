from src.utils import pmap
from src.paths import geometric_average, moment_average, alpha_average
from src.ais import Gaussian1D, find_alpha_average_batch, find_average_batch, run_ais
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

def compute_bounds_ais( proposal, target, no_betas,
    no_samples=10000,
    no_seeds=20,
    path="alpha",
    alpha=1.0,
    no_iters=10000,
):

    beta_vec = np.linspace(0, 1.0, no_betas)

    if path == "alpha":
        beta_dists = [alpha_average(proposal, target, beta, alpha) for beta in beta_vec]
    elif path == "moment":
        beta_dists = [moment_average(proposal, target, beta) for beta in beta_vec]
    else:
        beta_dists = [geometric_average(proposal, target, beta) for beta in beta_vec]

    def _run(seed):
        np.random.seed(seed)
        logZ, logZ_lower, logZ_upper = run_ais(beta_dists, no_samples)
        res = {
            "no_betas":no_betas,
            "seed":seed,
            "option":path,
            "alpha":alpha,
            "no_iters":no_iters,
            "logZ":logZ,
            "logZ_lower":logZ_lower,
            "logZ_upper":logZ_upper
        }

        return res

    return pd.DataFrame([_run(seed) for seed in range(no_seeds)])

def compare_bounds():
    alpha_vec = np.linspace(-1, 1, 21)
    no_betass = [3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 101]
    proposal = Gaussian1D(-4.0, 3)
    target   = Gaussian1D(4.0, 1)

    def _run_geom_moments(args):
        no_betas, path = args
        return compute_bounds_ais( proposal, target, no_betas, path = path)

    def _run_alpha(args):
        no_betas, alpha = args
        return compute_bounds_ais(proposal, target, no_betas, alpha = alpha, path="alpha")

    alpha = pmap(_run_alpha, product(no_betass, alpha_vec))
    geom_moments = pmap(_run_geom_moments, product(no_betass, ['geometric', 'moment']))
    df = pd.concat(geom_moments + alpha)

    return df

df = compare_bounds()
df.to_pickle("results/hmc.pkl")

