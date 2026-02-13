# NIFTY Volatility Surface & Monte Carlo Pricing Engine

A modular quantitative pricing engine for index options, featuring:

- Black–Scholes analytical pricing
- Monte Carlo simulation (GBM)
- Structured volatility modeling framework
- Extensible engine architecture

---

## Project Structure

src/
- black_scholes.py
- monte_carlo.py
- engine.py

examples/
- visualize_paths.py

tests/
- test_mc_convergence.py

---

## Installation

pip install -r requirements.txt

---

## Example

python examples/visualize_paths.py

---

## Validation

Monte Carlo prices converge to Black–Scholes benchmark within tolerance.

---

## Roadmap

- Implied volatility extraction
- Volatility smile modeling
- Term structure modeling
- Multi-expiry surface construction

---

## Author

Ayush Sharma
