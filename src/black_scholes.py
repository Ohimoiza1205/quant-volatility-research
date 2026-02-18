"""
Black-Scholes option pricing from scratch.

This module implements the closed-form Black-Scholes formula for European
call and put options. No external pricing libraries are used: only NumPy
and SciPy for the normal CDF. This is the baseline model against which the
Heston stochastic volatility model will be compared.

Reference:
    Black, F. and Scholes, M. (1973). The Pricing of Options and
    Corporate Liabilities. Journal of Political Economy, 81(3), 637-654.
"""

import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    """
    Compute the d1 term in the Black-Scholes formula.

    Parameters
    ----------
    S : float
        Current underlying asset price
    K : float
        Option strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annualized, continuously compounded)
    sigma : float
        Volatility of the underlying asset (annualized)

    Returns
    -------
    float
        The d1 value
    """
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    """
    Compute the d2 term in the Black-Scholes formula.
    d2 = d1 - sigma * sqrt(T)

    Parameters
    ----------
    Same as d1.

    Returns
    -------
    float
        The d2 value
    """
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    """
    Black-Scholes price for a European call option.

    The formula is: C = S * N(d1) - K * exp(-rT) * N(d2)
    where N() is the standard normal CDF.

    Parameters
    ----------
    S : float
        Current underlying asset price
    K : float
        Option strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annualized, continuously compounded)
    sigma : float
        Volatility of the underlying asset (annualized)

    Returns
    -------
    float
        Theoretical price of the European call option
    """
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    return S * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)


def put_price(S, K, T, r, sigma):
    """
    Black-Scholes price for a European put option.

    The formula is: P = K * exp(-rT) * N(-d2) - S * N(-d1)
    This follows directly from put-call parity.

    Parameters
    ----------
    Same as call_price.

    Returns
    -------
    float
        Theoretical price of the European put option
    """
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)


def put_call_parity_check(S, K, T, r, sigma):
    """
    Verify put-call parity: C - P = S - K * exp(-rT)

    Put-call parity is a fundamental no-arbitrage relationship. If it fails,
    there is an arbitrage opportunity. This function checks that our
    Black-Scholes implementation satisfies it.

    Parameters
    ----------
    Same as call_price.

    Returns
    -------
    float
        The difference (C - P) - (S - K * exp(-rT)); should be near zero
    """
    C = call_price(S, K, T, r, sigma)
    P = put_price(S, K, T, r, sigma)
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    return lhs - rhs


def delta_call(S, K, T, r, sigma):
    """
    Delta of a European call: dC/dS = N(d1)
    Measures sensitivity of call price to underlying price.
    """
    return norm.cdf(d1(S, K, T, r, sigma))


def delta_put(S, K, T, r, sigma):
    """
    Delta of a European put: dP/dS = N(d1) - 1
    For a put, delta is negative (price falls as underlying rises).
    """
    return norm.cdf(d1(S, K, T, r, sigma)) - 1


def gamma(S, K, T, r, sigma):
    """
    Gamma: d2C/dS2 = N'(d1) / (S * sigma * sqrt(T))
    Same for calls and puts. Measures convexity of option price.
    """
    _d1 = d1(S, K, T, r, sigma)
    return norm.pdf(_d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    Vega: dC/dsigma = S * N'(d1) * sqrt(T)
    Same for calls and puts. Measures sensitivity to volatility.
    Note: conventionally quoted per 1% change in vol, so divide by 100.
    """
    _d1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(_d1) * np.sqrt(T)


def implied_volatility(market_price, S, K, T, r, option_type="call", tol=1e-6, max_iter=200):
    """
    Compute implied volatility via bisection method.

    Given a market-observed option price, find the volatility sigma such that
    the Black-Scholes formula reproduces that price exactly. This is the
    inverse problem: rather than pricing given vol, we infer vol from price.

    Parameters
    ----------
    market_price : float
        Observed market price of the option
    S : float
        Current underlying price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free rate
    option_type : str
        "call" or "put"
    tol : float
        Convergence tolerance for bisection
    max_iter : int
        Maximum iterations

    Returns
    -------
    float
        Implied volatility, or np.nan if no solution found
    """
    pricing_func = call_price if option_type == "call" else put_price

    sigma_low = 1e-6
    sigma_high = 10.0

    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2.0
        price_mid = pricing_func(S, K, T, r, sigma_mid)

        if abs(price_mid - market_price) < tol:
            return sigma_mid

        if price_mid < market_price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid

    return np.nan


if __name__ == "__main__":
    # Sanity check with textbook example
    S = 100.0   # spot price
    K = 100.0   # at-the-money strike
    T = 1.0     # one year to expiration
    r = 0.05    # 5% risk-free rate
    sigma = 0.20  # 20% annual volatility

    C = call_price(S, K, T, r, sigma)
    P = put_price(S, K, T, r, sigma)
    parity_error = put_call_parity_check(S, K, T, r, sigma)

    print(f"Call price: {C:.4f}")
    print(f"Put price:  {P:.4f}")
    print(f"Put-call parity error: {parity_error:.2e}  (should be ~0)")
    print(f"Delta (call): {delta_call(S, K, T, r, sigma):.4f}")
    print(f"Gamma:        {gamma(S, K, T, r, sigma):.4f}")
    print(f"Vega:         {vega(S, K, T, r, sigma):.4f}")

    # Test implied vol roundtrip: price then recover vol
    IV = implied_volatility(C, S, K, T, r, option_type="call")
    print(f"Implied vol recovered: {IV:.4f}  (should be {sigma})")