import math
import numpy as np


def bernoulli_log_likelihood(data, theta):
    import math

def bernoulli_log_likelihood(data, theta):
    """
    Compute the Bernoulli log-likelihood for binary data.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    theta : float
        Bernoulli parameter, must satisfy 0 < theta < 1.

    Returns
    -------
    float
        Log-likelihood:
            sum_i [x_i log(theta) + (1-x_i) log(1-theta)]

    Requirements
    ------------
    - Raise ValueError if data is empty
    - Raise ValueError if theta is not in (0,1)
    - Raise ValueError if data contains values other than 0 and 1
    """
    if data is None or len(data) == 0:
        raise ValueError("data must not be empty")

    if not (0 < theta < 1):
        raise ValueError("theta must be in the interval (0, 1)")

    log_theta = math.log(theta)
    log_one_minus_theta = math.log(1 - theta)

    ll = 0.0
    for x in data:
        if x not in (0, 1):
            raise ValueError("data must contain only 0 and 1")
        ll += x * log_theta + (1 - x) * log_one_minus_theta

    return ll

    raise NotImplementedError("Implement bernoulli_log_likelihood")


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    """
    Estimate the Bernoulli MLE and compare candidate theta values.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    candidate_thetas : array-like or None
        Optional candidate theta values to compare using log-likelihood.
        If None, use [0.2, 0.5, 0.8].

    Returns
    -------
    dict
        A dictionary with:
        - 'mle': float
            The Bernoulli MLE
        - 'num_successes': int
        - 'num_failures': int
        - 'log_likelihoods': dict
            Mapping candidate theta -> log-likelihood
        - 'best_candidate': float
            Candidate theta with highest log-likelihood

    Requirements
    ------------
    - Validate data
    - Compute MLE analytically
    - Compute candidate log-likelihoods using bernoulli_log_likelihood
    - In case of ties in best candidate, return the first one encountered
    """
    # Validate data
    if data is None or len(data) == 0:
        raise ValueError("data must not be empty")

    for x in data:
        if x not in (0, 1):
            raise ValueError("data must contain only 0 and 1")

    # Default candidate thetas
    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    # Count successes and failures
    num_successes = sum(data)
    num_failures = len(data) - num_successes

    # Compute analytical MLE
    mle = num_successes / len(data)

    # Compute log-likelihoods for candidates
    log_likelihoods = {}
    best_candidate = None
    best_ll = None

    for theta in candidate_thetas:
        ll = bernoulli_log_likelihood(data, theta)
        log_likelihoods[theta] = ll

        if best_ll is None or ll > best_ll:
            best_ll = ll
            best_candidate = theta
        # ties: do nothing (keeps first encountered)

    return {
        "mle": mle,
        "num_successes": num_successes,
        "num_failures": num_failures,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate
    }
    raise NotImplementedError("Implement bernoulli_mle_with_comparison")


def poisson_log_likelihood(data, lam):
    """
    Compute the Poisson log-likelihood for count data.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    lam : float
        Poisson rate, must satisfy lam > 0.

    Returns
    -------
    float
        Log-likelihood:
            sum_i [x_i log(lam) - lam - log(x_i!)]

    Requirements
    ------------
    - Raise ValueError if data is empty
    - Raise ValueError if lam <= 0
    - Raise ValueError if data contains negative or non-integer values

    Notes
    -----
    You may use math.lgamma(x + 1) for log(x!) since log(x!) = lgamma(x+1).
    """
    if data is None or len(data) == 0:
        raise ValueError("data must not be empty")

    if lam <= 0:
        raise ValueError("lam must be > 0")

    ll = 0.0
    log_lam = math.log(lam)

    for x in data:
        if not isinstance(x, (int, np.integer)) or x < 0:
            raise ValueError("data must contain only nonnegative integers")

        ll += x * log_lam - lam - math.lgamma(x + 1)

    
    return ll

    raise NotImplementedError("Implement poisson_log_likelihood")


def poisson_mle_analysis(data, candidate_lambdas=None):
    """
    Estimate the Poisson MLE and compare candidate lambda values.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    candidate_lambdas : array-like or None
        Optional candidate lambdas to compare using log-likelihood.
        If None, use [1.0, 3.0, 5.0].

    Returns
    -------
    dict
        A dictionary with:
        - 'mle': float
            The Poisson MLE
        - 'sample_mean': float
        - 'total_count': int
        - 'n': int
        - 'log_likelihoods': dict
            Mapping candidate lambda -> log-likelihood
        - 'best_candidate': float
            Candidate lambda with highest log-likelihood

    Requirements
    ------------
    - Validate data
    - Compute MLE analytically
    - Compute candidate log-likelihoods using poisson_log_likelihood
    - In case of ties in best candidate, return the first one encountered
    """

    if data is None or len(data) == 0:
        raise ValueError("data must not be empty")

    for x in data:
        if not isinstance(x, (int, np.integer)) or x < 0:
            raise ValueError("data must contain only nonnegative integers")

    # Default candidate lambdas
    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    n = len(data)
    total_count = int(sum(data))
    sample_mean = total_count / n

    # Poisson MLE is sample mean
    mle = sample_mean

    # Compute log-likelihoods for candidates
    log_likelihoods = {}
    best_candidate = None
    best_ll = None

    for lam in candidate_lambdas:
        ll = poisson_log_likelihood(data, lam)
        log_likelihoods[lam] = ll

        if best_ll is None or ll > best_ll:
            best_ll = ll
            best_candidate = lam
        # tie -> keep first encountered

    return {
        "mle": mle,
        "sample_mean": sample_mean,
        "total_count": total_count,
        "n": n,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate
    }
    raise NotImplementedError("Implement poisson_mle_analysis")
