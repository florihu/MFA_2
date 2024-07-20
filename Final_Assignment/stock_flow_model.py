import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import weibull_min, norm, geom, uniform, lognorm
from scipy.linalg import toeplitz, solve


#%% Utility functions

def fixed_sf(n, lifetime):
    """Compute the survival function (SF) of a fixed lifetime distribution.
    The fixed lifetime distribution models a random variable that always
    takes on a fixed value of lifetime, after which it becomes 0.

    Args:
        n: The number of integers to evaluate the SF at.
        lifetime: fixed lifetime

    Returns:
        An array of the survival function (SF) values of the fixed lifetime
        distribution evaluated at integers from 0 to n-1.
    """
    pdf = np.zeros(n)
    pdf[0:lifetime] = 1
    return pdf


def weibull_sf(n, shape=5, loc=0, scale=20):
    """Compute the survival function (SF) of the Weibull distribution.

    Args:
        n: The number of integers to evaluate the SF at.
        shape: determine the skewness of the distribution
            (shape < 1 = skewed to the right, shape>1 = skewed to the left)
        loc: determine the minimum value that the distribution can take)
        scale: determine the spread of the distribution (scale = shift along x-axis)

    Returns:
        An array of the survival function (SF) values of the Weibull distribution
        evaluated at integers from 0 to n-1.
    """
    x = np.arange(n)
    return weibull_min.sf(x, c=shape, loc=loc, scale=scale)


def normal_sf(n, loc=30, scale=10):
    """Compute the survival function (SF) of the normal distribution.

    Args:
        n: Number of integers to evaluate the SF at.
        loc: Mean of the normal distribution.
        scale: Standard deviation of the normal distribution.

    Returns:
        An array of the survival function (SF) values of the normal distribution
        evaluated at integers from 0 to n-1.
    """
    x = np.arange(n)
    return norm.sf(x, loc=loc, scale=scale)


def geometric_sf(n, p=0.05, loc=0):
    """Compute the survival function (SF) of the geometric distribution. The geometric
    distribution models the number of trials that must be performed before the first success
    in a sequence of independent Bernoulli trials with the same probability of success.

    Args:
        n: The number of integers to evaluate the SF at.
        p: The probability of success for each Bernoulli trial.
        loc: shift along x-axis (i.e. the number of failures before the first success)

    Returns:
        An array of the survival function (SF) values of the geometric distribution
        evaluated at integers from 0 to n-1.
    """
    k = np.arange(n)
    return geom.sf(k=k, p=p, loc=loc)


def uniform_sf(n, loc=0, scale=10):
    """Compute the survival function (SF) of the continuous uniform distribution.
    The continuous uniform distribution models a random variable that can take
    any value in a fixed interval with equal probability.

    Args:
        n: The number of integers to evaluate the SF at.
        loc: The lower endpoint of the uniform distribution's support.
        scale: The width of the uniform distribution's support
            (i.e., the range between the lower and upper endpoints).

    Returns:
        An array of the survival function (SF) values of the uniform distribution
        evaluated at integers from 0 to n-1.
    """
    x = np.arange(n)
    return uniform.sf(x, loc=loc, scale=scale)


def lognormal_sf(n, x=0.01, loc=0, scale=1):
    """Compute the survival function (SF) of the log-normal distribution.
    The log-normal distribution models a random variable whose natural logarithm
    follows a normal distribution.

    Args:
        n: The number of integers to evaluate the SF at.
        x: The value at which to evaluate the SF.
        loc: The mean of the logarithm of the distribution.
        scale: The standard deviation of the logarithm of the distribution.

    Returns:
        An array of the survival function (SF) values of the log-normal
        distribution evaluated at integers from 0 to n-1.
    """
    s = np.arange(n)
    result = lognorm.sf(x, s, loc=loc, scale=scale)
    result[0] = 1
    return result


def compute_survival_curve(kind, n, **kwargs):
    """Compute the survival function (SF) for a given statistical distribution

    Args:
        kind: The kind of distribution to use. Current options are:
            [fixed, weibull, normal, lognormal, geometric, uniform]
        n: The number of integers to evaluate the SF at.
        **kwargs: Additional distribution-specific parameters.

    Returns:
        An array of the survival function (SF) values of the specified distribution
        evaluated at integers from 0 to n-1.
    """
    survival_function = {
        "fixed": fixed_sf,
        "weibull": weibull_sf,
        "normal": normal_sf,
        "geometric": geometric_sf,
        "uniform": uniform_sf,
        "lognormal": lognormal_sf,
    }
    return survival_function[kind](n, **kwargs)


def compute_toeplitz_sf(arr):
    """Compute the (lower triangle) Toeplitz matrix of a survival curve.

    Args:
        arr: An array representing the survival curve.

    Returns:
        The Toeplitz matrix of the survival curve
    """
    n = len(arr)
    return toeplitz(arr) * np.tril(np.ones(shape=(n, n)))


def convert_to_df(time, survival_curve, cohort, inflow, outflow, stock, nas):
    """Convert the different (numpy) arrays representing a stock-flow model into DataFrames.

    Args:
        time: the time axis of the model.
        survival_curve: the survival curve used for the model.
        cohort: the cohort data of the model.
        inflow: the inflow data of the model.
        outflow: the outflow data of the model.
        stock: the stock data of the model.
        nas: The net addition to stock data of the model.

    Returns:
        A tuple containing two Pandas DataFrames:
        `timeseries` representing 5 vectors (survival curve, inflow, outflow, nas and stock)
            with time as index
        `cohort` matrix (time as both index and columns)
    """
    timeseries = pd.DataFrame(
        data={
            "survival_curve": survival_curve,
            "inflow": inflow,
            "outflow": outflow,
            "stock": stock,
            "nas": nas,
        },
        index=time,
    )
    cohort = pd.DataFrame(cohort, index=time, columns=time)
    return timeseries, cohort

#%% Stock flow model

def flow_driven_model(time, inflow, sf_kind="normal", stock_ini=0, **kwargs):
    """Computes a stock-flow model given inflow

    Args:
        time (np.ndarray): 1D array of time values
        inflow (np.ndarray): 1D array of inflow values
        sf_kind (str): Type of survival function used to compute the survival curve (default 'normal').
            Valid options are 'fixed', 'weibull', 'normal', 'geometric', 'uniform', and 'lognormal'.
        stock_ini (float): Initial stock value (default 0)
        **kwargs: Additional keyword arguments for the selected survival function

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas DataFrames:
        - timeseries: A pandas DataFrame containing columns [survival_curve, inflow, outflow, stock, nas]
            and index as the given time array
        - cohort: A pandas DataFrame representing the cohort table, with rows and columns indexed by time.
            The values represent the number of individuals in the cohort at each time point.

    Raises:
        ValueError: If sf_kind is not a valid option
    """
    survival_curve = compute_survival_curve(kind=sf_kind, n=len(time), **kwargs)
    survival_toeplitz = compute_toeplitz_sf(survival_curve)
    cohort = inflow * survival_toeplitz
    stock = cohort.sum(axis=1)  # EXTRA STEP COMPARED WITH STOCK-DRIVEN
    nas = np.diff(stock, prepend=stock_ini)
    outflow = inflow - nas
    timeseries, cohort = convert_to_df(
        time, survival_curve, cohort, inflow, outflow, stock, nas
    )
    return timeseries, cohort


def stock_driven_model(time, stock, sf_kind="normal", stock_ini=0, **kwargs):
    """Computes a stock-flow model given stocks

    Args:
        time (np.ndarray): 1D array of time values
        stock (np.ndarray): 1D array of stock values
        sf_kind (str): Type of survival function used to compute the survival curve (default 'normal').
            Valid options are 'fixed', 'weibull', 'normal', 'geometric', 'uniform', and 'lognormal'.
        stock_ini (float): Initial stock value (default 0)
        **kwargs: Additional keyword arguments for the selected survival function

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas DataFrames:
        - timeseries: A pandas DataFrame containing columns [survival_curve, inflow, outflow, stock, nas]
            and index as the given time array
        - cohort: A pandas DataFrame representing the cohort table, with rows and columns indexed by time.
            The values represent the number of individuals in the cohort at each time point.

    Raises:
        ValueError: If sf_kind is not a valid option
    """
    survival_curve = compute_survival_curve(kind=sf_kind, n=len(time), **kwargs)
    survival_toeplitz = compute_toeplitz_sf(survival_curve)
    inflow = solve(survival_toeplitz, stock)  # EXTRA STEP COMPARED WITH FLOW-DRIVEN
    cohort = inflow * survival_toeplitz
    nas = np.diff(stock, prepend=stock_ini)
    outflow = inflow - nas
    timeseries, cohort = convert_to_df(
        time, survival_curve, cohort, inflow, outflow, stock, nas
    )
    return timeseries, cohort