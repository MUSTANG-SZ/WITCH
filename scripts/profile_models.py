"""Shared radial profile models for compare_xray_sz scripts."""

import numpy as np


def gNFW(r, p_0, r_s, alpha, beta, gamma):
    expression = ((r / r_s) ** -gamma) * (1 + (r / r_s) ** alpha) ** ((gamma - beta) / alpha)
    return p_0 * expression


def iso_beta(r, p_0, r_c, beta):
    expression = (1 + (r / r_c) ** 2) ** (-1.5 * beta)
    return p_0 * expression