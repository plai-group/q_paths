from src.hmc import HMCDist, gaus_log_prob
import autograd.numpy as np
import numpy as onp

def geometric_average(proposal, target, beta):
    assert 0 <= beta <= 1
    if beta == 0: return proposal
    pow1, pow2 = 1.0 - beta, beta

    def neg_logp(x):
        log_prob = proposal.logprob(x) * pow1 + target.logprob(x) * pow2
        return -log_prob

    return HMCDist(neg_logp)


def moment_average(proposal, target, beta):
    assert 0 <= beta <= 1
    if beta == 0: return proposal
    pow1, pow2 = 1.0 - beta, beta

    mean = pow1 * proposal.mean + pow2 * target.mean
    var = (
        pow1 * proposal.variance
        + pow2 * target.variance
        + pow1 * pow2 * (proposal.mean - target.mean) ** 2
    )
    var = onp.abs(var)  # TODO: fix small negative variance
    sig = onp.sqrt(var)

    def neg_logp(x):
        return -gaus_log_prob(x, mean, sig)

    return HMCDist(neg_logp)


def alpha_average(proposal, target, beta, alpha):
    assert 0 <= beta <= 1
    if beta == 0: return proposal
    pow1, pow2 = 1.0 - beta, beta

    def neg_logp(x):
        if alpha == 1:
            log_prob = pow1*proposal.logprob(x) + pow2*target.logprob(x)
        else:
            log_prob = (2/(1-alpha))*(np.logaddexp(
                np.log(pow1) + ((1-alpha)/2)* proposal.logprob(x),
                np.log(pow2) + ((1-alpha)/2)* target.logprob(x)))
        return -log_prob

    return HMCDist(neg_logp)
