"""QuantRisk Analytics - portfolio performance and risk analytics.

The package is split by responsibility so every calculation can be tested
without the user interface:

* ``data_loader``  - reading, validating and aligning price data
* ``sample_data``  - deterministic synthetic dataset used for offline demos
* ``portfolio``    - weight validation and portfolio return construction
* ``performance``  - return-based performance statistics
* ``risk_metrics`` - VaR / Expected Shortfall, rolling risk, diversification
* ``charts``       - Plotly figure builders (presentation only, no analytics)
"""

__version__ = "1.0.0"
