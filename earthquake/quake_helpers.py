""" Helper functions for earthquake.py

Estimators (all take station positions x (n, 2), delays t (n,), speed):
    centroid   average station position (the prompt's baseline)
    linear     ordinary least squares (OLS) on the circle equations, made
               linear by subtracting station 0's equation from the rest
    fit        nonlinear least squares on the delays themselves, started
               from several places: THE SUBMITTED ESTIMATOR
"""
import numpy as np
from scipy.optimize import least_squares
from earthquake import Earthquake


def centroid(x, t, speed):
    return x.mean(axis=0)


def linear(x, t, speed):
    """|x_i - y|^2 = r_i^2; subtracting station 0's equation makes it linear in y"""
    r = t * speed
    a = 2 * (x[1:] - x[0])
    b = (x[1:] ** 2).sum(1) - (x[0] ** 2).sum() - r[1:] ** 2 + r[0] ** 2
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _residual(y, x, t, speed):
    return np.linalg.norm(x - y, axis=1) / speed - t


def fit(x, t, speed):
    """least squares on delays, from several starts; keep the best fit"""
    starts = [linear(x, t, speed), centroid(x, t, speed), *x]
    sols = [least_squares(_residual, s, args=(x, t, speed)) for s in starts]
    return min(sols, key=lambda s: s.cost).x


ESTIMATORS = {'centroid': centroid, 'linear': linear, 'fit': fit}


def run(seeds, estimators=ESTIMATORS, **quake_kw):
    """ mse of each estimator on each seed's quake: dict name -> (len(seeds),) """
    mse = {name: [] for name in estimators}
    for seed in seeds:
        q = Earthquake(seed=seed, **quake_kw)
        for name, est in estimators.items():
            mse[name].append(q.evaluate(est(q.x, q.t, q.speed)))
    return {name: np.array(v) for name, v in mse.items()}


def summary(mse):
    """ one line per estimator: mean, median and 99th percentile mse, km^2 """
    for name, v in mse.items():
        print(f'{name:>12}: mean {v.mean():9.3f}   median {np.median(v):7.3f}'
              f'   99th pct {np.percentile(v, 99):9.3f}')


def sweep(param, values, seeds, names=('fit', 'posterior')):
    """ mean mse of each named estimator as one Earthquake argument varies """
    est = {k: ESTIMATORS[k] for k in names}
    out = {k: [] for k in names}
    for v in values:
        mse = run(seeds, est, **{param: v})
        for k in names:
            out[k].append(mse[k].mean())
    return {k: np.array(v) for k, v in out.items()}


def estimate(x, t, speed):
    """ the submitted estimator (leaderboard signature) """
    return fit(np.asarray(x, float), np.asarray(t, float), speed)


def plot_quake(ax, quake, estimates, half_width=None):
    """each station's span and the estimates on one map

    Args:
        ax: matplotlib axes
        quake: only its x, t and speed are used
        estimates (dict): label -> (2,) estimated epicenter, km
        half_width (float): zoom to +/- this many km around the
            last estimate
    """
    import matplotlib.pyplot as plt
    for xi, ri in zip(quake.x, quake.t * quake.speed):
        ax.add_patch(plt.Circle(xi, ri, fill=False, color='gray', lw=.7))
    ax.plot(*quake.x.T, 'k^', label='station')
    for (label, y_hat), m in zip(estimates.items(), 'sDo'):
        ax.plot(*y_hat, m, ms=8, label=label)
    if half_width is not None:
        ax.set_xlim(y_hat[0] - half_width, y_hat[0] + half_width)
        ax.set_ylim(y_hat[1] - half_width, y_hat[1] + half_width)
    ax.set_aspect('equal')
    ax.set(xlabel='km east', ylabel='km north')
