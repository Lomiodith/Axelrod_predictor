"""Reusable WTI crude-oil forecasting package.

Ported from the exploratory notebooks:

* ``WTI_Direction_Model.ipynb``  -> horizon ``hourly``  (classification: next hour up or down)
* ``WTI_Price_Model.ipynb``      -> horizon ``daily``   (regression: tomorrow's % change)

Run it from the terminal::

    python -m wti train   --horizon daily
    python -m wti predict --horizon hourly
"""

from wti.config import DAILY, HORIZONS, HOURLY, Horizon

__all__ = ["DAILY", "HOURLY", "HORIZONS", "Horizon"]
__version__ = "0.1.0"
