from src.utils import pmap
from src.paths import geometric_average, moment_average, alpha_average
from src.ais import Gaussian1D, find_alpha_average_batch, find_average_batch, run_ais
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_bounds_ais( proposal, target, no_betas,
    no_samples = 1000,
    no_seeds=20,
    option="alpha",
    alpha=1.0,
    no_iters=10000,
):

    beta_vec = np.linspace(0, 1.0, no_betas)

    if option == "alpha":
        # if use alpha averaging
        beta_dists = find_alpha_average_batch(proposal, target, beta_vec, [alpha], no_iters)[0]
    else:
        # if use moment or geometric averaging
        beta_dists = find_average_batch(proposal, target, beta_vec, option)

    def _run(seed):
        np.random.seed(seed)
        logZ_lower, logZ_upper = run_ais(beta_dists, no_samples)
        res = {
            "no_betas":no_betas,
            "seed":seed,
            "option":option,
            "alpha":alpha,
            "no_iters":no_iters,
            "logZ_lower":logZ_lower,
            "logZ_upper":logZ_upper
        }

        return res

    return pd.DataFrame([_run(seed) for seed in range(no_seeds)])

def compare_bounds():
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 1.0]
    no_betass = [3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 101]
    proposal = Gaussian1D(-4.0, 0.2)
    target   = Gaussian1D(4.0, 1.0)

    def _inner(no_betas):
        geometric = compute_bounds_ais( proposal, target, no_betas, option = "geometric")

        # run moment
        moment = compute_bounds_ais( proposal, target, no_betas, option = "moment")

        # run alpha
        alpha = pd.concat([compute_bounds_ais(proposal, target, no_betas, alpha = a, option="alpha") for a in alphas])

        return pd.concat([geometric , moment, alpha])


    df = pd.concat(pmap(_inner, no_betass))

    return df

df = compare_bounds()
df.to_pickle("results/baseline.pkl")
