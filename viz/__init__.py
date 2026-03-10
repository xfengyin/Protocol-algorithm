# Visualization Module for Protocol-algorithm

"""
Protocol-algorithm Visualization

This module provides beautiful visualizations for WSN protocol simulations.

Features:
- Matplotlib static plots (publication quality)
- Plotly interactive charts (web-ready)
- Modern color schemes
- Multiple export formats (PNG, SVG, HTML, PDF)

Usage:
    from viz.demo_viz import plot_network, plot_metrics
    
    plot_network(positions, cluster_heads)
    plot_metrics(results)
"""

from .demo_viz import (
    plot_network_matplotlib,
    plot_network_plotly,
    plot_energy_curve,
    COLORS,
)

__all__ = [
    'plot_network_matplotlib',
    'plot_network_plotly',
    'plot_energy_curve',
    'COLORS',
]

__version__ = '2.0.0'
