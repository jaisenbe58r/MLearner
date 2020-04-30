"""
Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import chart_studio.plotly as py
import plotly.graph_objects as go
import chart_studio
from plotly.subplots import make_subplots

from .ml_plotly import plotly_histogram2, plot_PCA, plot_LDA, FeatureAnalyst

chart_studio.tools.set_credentials_file(username='jaisenbe58r', api_key='VKTDdiRq6IFfTHG0dYoZ')


__all__ = ["plotly_histogram2","plot_PCA", "plot_LDA", "FeatureAnalyst"]
