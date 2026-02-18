# Quant Volatility Research

## Neural Network Calibration of the Heston Stochastic Volatility Model

This project implements and compares two approaches to calibrating the Heston stochastic
volatility model to real SPX options market data: classical least-squares optimization
and a neural network surrogate model.

## What This Project Does

The Heston model (1993) prices European options by modeling volatility as a
mean-reverting stochastic process rather than a constant. Calibrating the model to
observed market prices requires solving an optimization problem that is slow under
classical methods. This project trains a neural network to approximate the Heston
pricing function and uses it as a fast oracle for calibration.

## Project Status

Week 1: Black-Scholes implementation from scratch (baseline model).

## Repository Structure

notebooks/ - Jupyter notebooks documenting each stage of the project
src/ - Production Python modules
data/ - SPX options data collected via yfinance
paper/ - Final write-up

## Dependencies

See requirements.txt

## Background Reading

Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
Journal of Political Economy, 81(3), 637-654.

Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility
with Applications to Bond and Currency Options.
Review of Financial Studies, 6(2), 327-343.

Horvath, B., Muguruza, A., and Tomas, M. (2021). Deep Learning Volatility.
Quantitative Finance, 21(1), 11-27.
