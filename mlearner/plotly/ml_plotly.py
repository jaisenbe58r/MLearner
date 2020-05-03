
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import chart_studio.plotly as py
import plotly.graph_objects as go
import chart_studio
import matplotlib.pyplot as plt

from mlearner import preprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sk_pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

chart_studio.tools.set_credentials_file(username='jaisenbe58r', api_key='VKTDdiRq6IFfTHG0dYoZ')


def plotly_histogram2(X, columns, target):
    colors = {2: 'rgb(255,127,20)',
              3: 'rgb(31, 220, 120)',
              13: 'rgb(44, 50, 180)'}
    traces = []
    _targets = sorted(X[target].unique().tolist())

    legend = {2: True, 3: True, 13: True}

    for col in range(2):
        for key in range(len(_targets)):
            traces.append(go.Histogram(x=X[X[target] == _targets[key]][columns[col]], opacity=0.7, 
                                        xaxis="x%s" % (col+1), marker=go.Marker(color=colors[_targets[key]]),
                                        name=_targets[key], showlegend=legend[_targets[key]]))
        legend = {2: False, 3: False, 13: False}

    data = go.Data(traces)
    layout = go.Layout(barmode="overlay",
                        xaxis=go.XAxis(domain=[0, 0.48], title=columns[0]),
                        xaxis2=go.XAxis(domain=[0.52, 1], title=columns[1]),
                        yaxis=go.YAxis(title="Número de Defectos"),
                        title="Histograma caracteristicas")

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)

    return fig


def plot_PCA(data, features):
    X = data[features]
    y = data["categoria"]

    X_std = StandardScaler().fit_transform(X)
    acp = sk_pca(n_components=2)
    Y = acp.fit_transform(X_std)

    results = []

    for name in (2, 3, 13):
        result = go.Scatter(x=Y[y == name, 0], y=Y[y == name, 1],
                            mode="markers", name=name,
                            marker=go.Marker(size=8, line=go.Line(color="rgba(225,225,225,0.2)", width=0.5),
                                                opacity=0.75))
        results.append(result)

    data = go.Data(results)
    layout = go.Layout(xaxis=go.XAxis(title="CP1", showline=False), 
                        yaxis=go.YAxis(title="CP2", showline=False))

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)

    return fig


def plot_LDA(data, features):
    X = data[features]
    y = data["categoria"]

    X_std = StandardScaler().fit_transform(X)
    LDA = LinearDiscriminantAnalysis()
    Y = LDA.fit_transform(X_std, y)

    results = []

    for name in (2, 3, 13):
        result = go.Scatter(x=Y[y == name, 0], y=Y[y == name, 1],
                            mode="markers", name=name,
                            marker=go.Marker(size=8, line=go.Line(color="rgba(225,225,225,0.2)", width=0.5),
                                                opacity=0.75))
        results.append(result)

    data = go.Data(results)
    layout = go.Layout(xaxis=go.XAxis(title="CP1", showline=False), 
                        yaxis=go.YAxis(title="CP2", showline=False))

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)

    return fig


def FeatureAnalyst(X, feature, target, targets=[2, 3, 13]):
    """Analisis de la caracteristica respecto a las categorias."""
    _targets = sorted(X[target].unique())
    data = preprocessing.ExtractCategories(targets, target=["categoria"]).fit_transform(X)

    figure, axs = plt.subplots(1, 2, figsize=(16, 5))
    ax = axs.flatten()

    # Histograma - Datos brutos
    _data = data
    for i in _targets:
        ax[0].hist(_data[_data[target] == i][feature], alpha=0.7, density=True)
    ax[0].set_title("Histograma - Datos brutos")
    ax[0].legend(_targets)

    # Histograma - Escala Logaritmica
    _data = preprocessing.FixSkewness(columns=[feature], drop=False).fit_transform(data)
    for i in _targets:
        ax[1].hist(_data[_data[target] == i][feature], alpha=0.7, density=True)
    ax[1].set_title("Histograma - Escala Logaritmica")
    ax[1].legend(_targets)
