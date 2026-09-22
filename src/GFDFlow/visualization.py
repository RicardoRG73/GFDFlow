"""
Visualization module for GFDFlow.

Provides standardized plotting functions for 2D and 3D solutions, node distributions,
normal vectors, and phreatic surfaces using Matplotlib.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt


def _ensure_ax_2d(
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> Tuple[Figure, Axes]:
    """Return an existing or new 2D figure and axis."""
    if ax is None:
        fig, new_ax = plt.subplots(figsize=figsize)
        return fig, new_ax
    return ax.figure, ax


def _ensure_ax_3d(
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> Tuple[Figure, Axes]:
    """Return an existing or new 3D figure and axis."""
    if ax is None:
        fig = plt.figure(figsize=figsize)
        new_ax = fig.add_subplot(111, projection="3d")
        return fig, new_ax
    return ax.figure, ax


def _save_fig_if_needed(fig: Figure, savepath: Optional[str] = None, dpi: int = 300) -> None:
    """Save the figure to disk if savepath is provided."""
    if savepath:
        dirname = os.path.dirname(savepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")


def plot_solution_2d(
    coords: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    triangles: Optional[npt.NDArray[np.int_]] = None,
    levels: Union[int, npt.ArrayLike] = 20,
    cmap: str = "plasma",
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    contour_lines: bool = True,
    line_colors: str = "k",
    linewidths: float = 0.5,
    line_alpha: float = 0.6,
    clabel: bool = False,
    overlay_nodes: Optional[Union[Dict[str, npt.ArrayLike], Iterable[Tuple[str, npt.ArrayLike]], npt.ArrayLike]] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    xlabel: str = "x",
    ylabel: str = "y",
    equal_aspect: bool = True,
    savepath: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """Plot a 2D contour map (tricontourf) of a scalar field on arbitrary node coordinates.

    Parameters
    ----------
    coords : npt.NDArray[np.float64]
        Array of shape (N, 2) containing node coordinates [x, y].
    u : npt.NDArray[np.float64]
        Array of shape (N,) with the solution values at each node.
    triangles : Optional[npt.NDArray[np.int_]], default=None
        Optional triangle connectivity matrix of shape (M, 3). If None, Delaunay
        triangulation is performed internally by Matplotlib.
    levels : int or array-like, default=20
        Number or sequence of contour levels.
    cmap : str, default='plasma'
        Colormap name.
    colorbar : bool, default=True
        Whether to draw a colorbar.
    colorbar_label : Optional[str], default=None
        Label text for the colorbar.
    contour_lines : bool, default=True
        Whether to draw contour lines over the filled contours.
    line_colors : str, default='k'
        Color of the contour lines.
    linewidths : float, default=0.5
        Width of the contour lines.
    line_alpha : float, default=0.6
        Alpha transparency for contour lines.
    clabel : bool, default=False
        Whether to show numeric labels along contour lines.
    overlay_nodes : Optional[Union[dict, iterable, array-like]], default=None
        Nodes to highlight over the contour (e.g. interfaces, boundaries).
        Can be a dict ``{label: node_indices}``, an iterable of ``(label, node_indices)``,
        or a single array of indices.
    ax : Optional[Axes], default=None
        Existing Matplotlib Axes to draw onto. If None, a new figure and axes are created.
    figsize : Tuple[float, float], default=(8, 6)
        Figure size in inches (used if ax is None).
    title : Optional[str], default=None
        Title of the plot.
    xlabel : str, default='x'
        Label for the X-axis.
    ylabel : str, default='y'
        Label for the Y-axis.
    equal_aspect : bool, default=True
        Whether to set equal aspect ratio.
    savepath : Optional[str], default=None
        File path to save the figure image.
    **kwargs : Any
        Additional keyword arguments passed to ``ax.tricontourf``.

    Returns
    -------
    Tuple[Figure, Axes]
        The Matplotlib Figure and Axes objects.
    """
    fig, ax = _ensure_ax_2d(ax, figsize=figsize)

    x = coords[:, 0]
    y = coords[:, 1]

    if triangles is not None:
        tcf = ax.tricontourf(x, y, triangles, u, levels=levels, cmap=cmap, **kwargs)
        if contour_lines:
            tcl = ax.tricontour(
                x, y, triangles, u, levels=levels, colors=line_colors, linewidths=linewidths, alpha=line_alpha
            )
            if clabel:
                ax.clabel(tcl, inline=True)
    else:
        tcf = ax.tricontourf(x, y, u, levels=levels, cmap=cmap, **kwargs)
        if contour_lines:
            tcl = ax.tricontour(
                x, y, u, levels=levels, colors=line_colors, linewidths=linewidths, alpha=line_alpha
            )
            if clabel:
                ax.clabel(tcl, inline=True)

    if colorbar:
        cbar = fig.colorbar(tcf, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    # Highlight overlay nodes if given
    if overlay_nodes is not None:
        if isinstance(overlay_nodes, dict):
            for label, node_idx in overlay_nodes.items():
                idx = np.asarray(node_idx, dtype=int).ravel()
                ax.scatter(x[idx], y[idx], s=15, label=label, alpha=0.8)
            ax.legend()
        elif isinstance(overlay_nodes, (list, tuple)) and len(overlay_nodes) > 0 and isinstance(overlay_nodes[0], tuple):
            for item in overlay_nodes:
                label, node_idx = item[0], item[1]
                idx = np.asarray(node_idx, dtype=int).ravel()
                ax.scatter(x[idx], y[idx], s=15, label=label, alpha=0.8)
            ax.legend()
        else:
            idx = np.asarray(overlay_nodes, dtype=int).ravel()
            ax.scatter(x[idx], y[idx], s=15, c="k", alpha=0.5)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if equal_aspect:
        ax.set_aspect("equal")

    _save_fig_if_needed(fig, savepath)
    return fig, ax


def plot_solution_3d(
    coords: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    triangles: Optional[npt.NDArray[np.int_]] = None,
    cmap: str = "plasma",
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    view_init: Optional[Tuple[float, float]] = (30, -120),
    edge_color: Optional[str] = None,
    alpha: float = 1.0,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "U",
    savepath: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """Plot a 3D surface (plot_trisurf) of a scalar field on node coordinates.

    Parameters
    ----------
    coords : npt.NDArray[np.float64]
        Array of shape (N, 2) containing node coordinates [x, y].
    u : npt.NDArray[np.float64]
        Array of shape (N,) with the solution values at each node.
    triangles : Optional[npt.NDArray[np.int_]], default=None
        Optional triangle connectivity matrix of shape (M, 3).
    cmap : str, default='plasma'
        Colormap name.
    colorbar : bool, default=True
        Whether to show a colorbar.
    colorbar_label : Optional[str], default=None
        Label text for the colorbar.
    view_init : Optional[Tuple[float, float]], default=(30, -120)
        Elevation and azimuth angles (elev, azim) for 3D camera.
    edge_color : Optional[str], default=None
        Color for triangle edges on the 3D surface.
    alpha : float, default=1.0
        Surface opacity.
    ax : Optional[Axes], default=None
        Existing 3D Axes to draw onto.
    figsize : Tuple[float, float], default=(8, 6)
        Figure size in inches.
    title : Optional[str], default=None
        Title of the plot.
    xlabel, ylabel, zlabel : str, default='x', 'y', 'U'
        Axis labels.
    savepath : Optional[str], default=None
        File path to save the figure image.
    **kwargs : Any
        Additional keyword arguments passed to ``ax.plot_trisurf``.

    Returns
    -------
    Tuple[Figure, Axes]
        The Matplotlib Figure and Axes objects.
    """
    fig, ax = _ensure_ax_3d(ax, figsize=figsize)

    x = coords[:, 0]
    y = coords[:, 1]

    surf_kwargs: Dict[str, Any] = {"cmap": cmap, "alpha": alpha, "antialiased": False}
    if edge_color is not None:
        surf_kwargs["edgecolor"] = edge_color
    if triangles is not None:
        surf_kwargs["triangles"] = triangles
    surf_kwargs.update(kwargs)

    surf = ax.plot_trisurf(x, y, u, **surf_kwargs)

    if colorbar:
        cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=15)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    if view_init is not None:
        ax.view_init(elev=view_init[0], azim=view_init[1])

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zlabel:
        ax.set_zlabel(zlabel)

    _save_fig_if_needed(fig, savepath)
    return fig, ax


def plot_solution_comparison_3d(
    coords: npt.NDArray[np.float64],
    u_num: npt.NDArray[np.float64],
    u_exact: npt.NDArray[np.float64],
    triangles: Optional[npt.NDArray[np.int_]] = None,
    num_label: str = "Numerical",
    exact_label: str = "Exact",
    num_color: str = "r",
    exact_color: str = "b",
    alpha: float = 0.5,
    view_init: Optional[Tuple[float, float]] = (20, -50),
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "U",
    savepath: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """Overlay numerical and exact 3D surfaces for comparison.

    Parameters
    ----------
    coords : npt.NDArray[np.float64]
        Array of shape (N, 2) containing node coordinates [x, y].
    u_num : npt.NDArray[np.float64]
        Numerical solution field of shape (N,).
    u_exact : npt.NDArray[np.float64]
        Exact/analytical solution field of shape (N,).
    triangles : Optional[npt.NDArray[np.int_]], default=None
        Triangle connectivity matrix.
    num_label : str, default='Numerical'
        Legend label for numerical solution.
    exact_label : str, default='Exact'
        Legend label for exact solution.
    num_color : str, default='r'
        Surface color for numerical solution.
    exact_color : str, default='b'
        Surface color for exact solution.
    alpha : float, default=0.5
        Transparency for overlapping surfaces.
    view_init : Optional[Tuple[float, float]], default=(20, -50)
        Elevation and azimuth camera angles.
    ax : Optional[Axes], default=None
        Existing 3D Axes.
    figsize : Tuple[float, float], default=(8, 6)
        Figure dimensions.
    title : Optional[str], default=None
        Title.
    xlabel, ylabel, zlabel : str, default='x', 'y', 'U'
        Axis labels.
    savepath : Optional[str], default=None
        Path to save figure.

    Returns
    -------
    Tuple[Figure, Axes]
        The Matplotlib Figure and Axes objects.
    """
    fig, ax = _ensure_ax_3d(ax, figsize=figsize)

    x = coords[:, 0]
    y = coords[:, 1]

    tri_kw = {"triangles": triangles} if triangles is not None else {}

    ax.plot_trisurf(
        x, y, u_num, color=num_color, alpha=alpha, label=num_label, antialiased=False, **tri_kw, **kwargs
    )
    ax.plot_trisurf(
        x, y, u_exact, color=exact_color, alpha=alpha, label=exact_label, antialiased=False, **tri_kw, **kwargs
    )

    ax.legend()
    if view_init is not None:
        ax.view_init(elev=view_init[0], azim=view_init[1])

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zlabel:
        ax.set_zlabel(zlabel)

    _save_fig_if_needed(fig, savepath)
    return fig, ax


def plot_nodes(
    coords: npt.NDArray[np.float64],
    node_groups: Union[Dict[str, npt.ArrayLike], Iterable[Tuple[str, npt.ArrayLike]]],
    point_size: float = 20.0,
    alpha: float = 0.7,
    legend: bool = True,
    legend_bbox: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = "Nodes Distribution",
    xlabel: str = "x",
    ylabel: str = "y",
    equal_aspect: bool = True,
    savepath: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """Plot classified node groups (materials, boundaries, interfaces) on 2D coordinates.

    Parameters
    ----------
    coords : npt.NDArray[np.float64]
        Array of shape (N, 2) containing node coordinates [x, y].
    node_groups : Union[Dict[str, ArrayLike], Iterable[Tuple[str, ArrayLike]]]
        Mapping or sequence of ``(label, node_indices)`` representing node sets.
    point_size : float, default=20.0
        Scatter marker size.
    alpha : float, default=0.7
        Point transparency.
    legend : bool, default=True
        Whether to show the legend.
    legend_bbox : Optional[Tuple[float, float]], default=None
        Bounding box anchor for legend placement, e.g. ``(1.05, 1)``.
    ax : Optional[Axes], default=None
        Existing Matplotlib Axes.
    figsize : Tuple[float, float], default=(8, 6)
        Figure dimensions.
    title : Optional[str], default='Nodes Distribution'
        Plot title.
    xlabel, ylabel : str, default='x', 'y'
        Axis labels.
    equal_aspect : bool, default=True
        Set equal aspect ratio.
    savepath : Optional[str], default=None
        Path to save figure.

    Returns
    -------
    Tuple[Figure, Axes]
        The Matplotlib Figure and Axes objects.
    """
    fig, ax = _ensure_ax_2d(ax, figsize=figsize)

    x = coords[:, 0]
    y = coords[:, 1]

    items = node_groups.items() if isinstance(node_groups, dict) else node_groups
    for label, node_indices in items:
        idx = np.asarray(node_indices, dtype=int).ravel()
        if idx.size == 0:
            continue
        ax.scatter(x[idx], y[idx], s=point_size, label=label, alpha=alpha, **kwargs)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if equal_aspect:
        ax.set_aspect("equal")

    if legend:
        if legend_bbox is not None:
            ax.legend(bbox_to_anchor=legend_bbox, loc="upper left")
        else:
            ax.legend(loc="best")

    _save_fig_if_needed(fig, savepath)
    return fig, ax


def plot_normal_vectors(
    coords: npt.NDArray[np.float64],
    normal_vecs: npt.NDArray[np.float64],
    boundary_nodes: Optional[Union[npt.ArrayLike, Iterable[npt.ArrayLike]]] = None,
    scatter: bool = True,
    point_size: float = 15.0,
    quiver_color: str = "k",
    quiver_alpha: float = 0.5,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = "Normal Vectors",
    xlabel: str = "x",
    ylabel: str = "y",
    equal_aspect: bool = True,
    savepath: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """Plot outward normal vectors at boundary or interface nodes using quiver.

    Parameters
    ----------
    coords : npt.NDArray[np.float64]
        Array of shape (N, 2) containing node coordinates [x, y].
    normal_vecs : npt.NDArray[np.float64]
        Array of shape (N, 2) containing unit normal vectors.
    boundary_nodes : Optional[Union[ArrayLike, Iterable[ArrayLike]]], default=None
        Node indices to plot normal vectors for. If None, normal vectors are plotted
        for all nodes where norm is nonzero (> 1e-6).
    scatter : bool, default=True
        Whether to scatter-plot the node points at vector origins.
    point_size : float, default=15.0
        Scatter marker size.
    quiver_color : str, default='k'
        Color of the quiver arrows.
    quiver_alpha : float, default=0.5
        Transparency of quiver arrows.
    ax : Optional[Axes], default=None
        Existing Matplotlib Axes.
    figsize : Tuple[float, float], default=(8, 6)
        Figure dimensions.
    title : Optional[str], default='Normal Vectors'
        Plot title.
    xlabel, ylabel : str, default='x', 'y'
        Axis labels.
    equal_aspect : bool, default=True
        Set equal aspect ratio.
    savepath : Optional[str], default=None
        Path to save figure.
    **kwargs : Any
        Additional keyword arguments passed to ``ax.quiver``.

    Returns
    -------
    Tuple[Figure, Axes]
        The Matplotlib Figure and Axes objects.
    """
    fig, ax = _ensure_ax_2d(ax, figsize=figsize)

    # Determine nodes to draw
    if boundary_nodes is None:
        norms = np.linalg.norm(normal_vecs, axis=1)
        nodes_list = [np.where(norms > 1e-6)[0]]
    elif isinstance(boundary_nodes, (list, tuple)) and len(boundary_nodes) > 0 and isinstance(boundary_nodes[0], (list, tuple, np.ndarray)):
        nodes_list = [np.asarray(b, dtype=int).ravel() for b in boundary_nodes]
    else:
        nodes_list = [np.asarray(boundary_nodes, dtype=int).ravel()]

    all_nodes = np.concatenate(nodes_list) if nodes_list else np.array([], dtype=int)

    if scatter and all_nodes.size > 0:
        ax.scatter(coords[all_nodes, 0], coords[all_nodes, 1], s=point_size, color=quiver_color, alpha=quiver_alpha)

    for b in nodes_list:
        if b.size == 0:
            continue
        ax.quiver(
            coords[b, 0],
            coords[b, 1],
            normal_vecs[b, 0],
            normal_vecs[b, 1],
            color=quiver_color,
            alpha=quiver_alpha,
            **kwargs,
        )

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if equal_aspect:
        ax.set_aspect("equal")

    _save_fig_if_needed(fig, savepath)
    return fig, ax


def plot_phreatic_surface(
    ax: Axes,
    coords: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    triangles: Optional[npt.NDArray[np.int_]] = None,
    density_g: float = 9.81,
    level: float = 0.0,
    color: str = "b",
    linewidths: float = 2.0,
    label: Optional[str] = "Phreatic surface",
    **kwargs: Any,
) -> None:
    """Superimpose a phreatic surface contour (pore pressure = 0) on an existing 2D plot.

    Computes the pressure head $(u - y) \cdot g$ and draws the zero contour level.

    Parameters
    ----------
    ax : Axes
        The Matplotlib Axes on which to draw.
    coords : npt.NDArray[np.float64]
        Array of shape (N, 2) containing node coordinates [x, y].
    u : npt.NDArray[np.float64]
        Total hydraulic head values at each node of shape (N,).
    triangles : Optional[npt.NDArray[np.int_]], default=None
        Optional triangle connectivity.
    density_g : float, default=9.81
        Unit weight of water ($\gamma = \rho \cdot g$).
    level : float, default=0.0
        Pressure contour level representing phreatic surface.
    color : str, default='b'
        Line color.
    linewidths : float, default=2.0
        Line width.
    label : Optional[str], default='Phreatic surface'
        Label for legend.
    **kwargs : Any
        Additional keyword arguments passed to ``ax.tricontour``.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    pressure = (u - y) * density_g

    if triangles is not None:
        cs = ax.tricontour(
            x, y, triangles, pressure, levels=[level], colors=color, linewidths=linewidths, **kwargs
        )
    else:
        cs = ax.tricontour(
            x, y, pressure, levels=[level], colors=color, linewidths=linewidths, **kwargs
        )

    if label:
        # Add proxy artist for legend support since ContourSet does not automatically show in legend
        from matplotlib.lines import Line2D
        line_proxy = Line2D([0], [0], color=color, linewidth=linewidths, label=label)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(line_proxy)
        labels.append(label)
        ax.legend(handles=handles, labels=labels)
