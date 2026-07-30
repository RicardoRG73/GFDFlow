
import numpy as np
import scipy.sparse as sp
from typing import Callable, Dict, List, Tuple, Union, Optional
import numpy.typing as npt

def get_support_nodes(
    node_idx: int,
    triangles: npt.NDArray[np.int_],
    min_support_nodes: int = 5,
    max_iter: int = 2
) -> npt.NDArray[np.int_]:
    """
    Returns the index of support nodes `I` corresponding to the central node
    with index `node_idx`.

    Parameters
    ----------
    node_idx : int
        index of central node.
    triangles : npt.NDArray[np.int_]
        array with shape (n,3), containing index of the n triangles with 3 nodes each.
    min_support_nodes : int, optional
        number of minimum support nodes. The default is 5.
    max_iter : int, optional
        number of maximum iterations for adding support nodes to the list `I`. The default is 2.

    Returns
    -------
    support_nodes : npt.NDArray[np.int_]
        index of the support nodes of central `node_idx`.
    """
    support_nodes = {node_idx}  # Use a set for unique support nodes
    iter_count = 0

    while len(support_nodes) < min_support_nodes and iter_count < max_iter:
        # Find triangles containing the current support nodes
        temp = np.any(np.isin(
            triangles,
            list(support_nodes)
        ), axis=1)
        support_nodes.update(triangles[temp].flatten())  # Add new nodes to the set
        iter_count += 1

    return np.array(list(support_nodes))

def compute_normal_vectors(
    boundary_nodes: npt.NDArray[np.int_],
    coords: npt.NDArray[np.float64],
    line_tolerance: float = 0.99
) -> npt.NDArray[np.float64]:
    """
    Computes outward normal vectors at boundary nodes.

    Automatically detects whether the boundary is a straight line. If it is,
    all nodes share the same uniform normal (fast and exact). Otherwise, a
    per-node normal is computed from the tangent defined by the 3 nearest
    boundary neighbours, and the sign is chosen so that the vector points
    away from the mesh centroid.

    Parameters
    ----------
    boundary_nodes : npt.NDArray[np.int_]
        Indices of the boundary nodes for this boundary segment.
    coords : npt.NDArray[np.float64]
        Array with shape (n, 2) containing the coordinates of all n nodes.
    all_coords : npt.NDArray[np.float64], optional
        Coordinates of *all* mesh nodes, used to compute the global centroid
        for the outward-sign check on curved boundaries. Defaults to ``coords``
        when not provided.
    line_tolerance : float, optional
        Dot-product threshold above which the boundary is classified as a
        straight line. Default is 0.99.

    Returns
    -------
    normal_vecs : npt.NDArray[np.float64]
        Array with shape (N, 2) containing the unit normal vectors at the
        N boundary nodes.
    """
    N = boundary_nodes.shape[0]

    if N < 2:
        return np.zeros((N, 2))

    clockwise_rotation = np.array([[0, 1], [-1, 0]])

    # --- Straight-line detection ---
    line_1 = coords[boundary_nodes[1]] - coords[boundary_nodes[0]]
    norm_1 = np.linalg.norm(line_1)
    if norm_1 > 0:
        line_1 = line_1 / norm_1

    line_2 = coords[boundary_nodes[N // 2]] - coords[boundary_nodes[0]]
    norm_2 = np.linalg.norm(line_2)
    if norm_2 > 0:
        line_2 = line_2 / norm_2

    if np.dot(line_1, line_2) > line_tolerance:
        # Straight boundary: uniform normal for all nodes
        line_normal = clockwise_rotation @ line_1
        return np.tile(line_normal, (N, 1))

    # --- Curved boundary: per-node normal via 3-nearest-neighbour tangent ---
    if N < 3:
        return np.zeros((N, 2))

    # Global centroid used to enforce outward orientation
    global_centroid = np.mean(coords, axis=0)

    normal_vecs = np.zeros((N, 2))
    for i, node in enumerate(boundary_nodes):
        distance = np.sqrt(
            (coords[node, 0] - coords[boundary_nodes, 0]) ** 2
            + (coords[node, 1] - coords[boundary_nodes, 1]) ** 2
        )

        closest_nodes = boundary_nodes[distance.argsort()[:3]]

        # Tangent: from first to last of the 3 closest nodes
        diff_v = coords[closest_nodes[2]] - coords[closest_nodes[0]]
        norm_diff = np.linalg.norm(diff_v)

        if norm_diff > 1e-6:
            ni = clockwise_rotation @ (diff_v / norm_diff)
        else:
            # Degenerate case: fall back to radial direction
            radial = coords[node] - global_centroid
            r_norm = np.linalg.norm(radial)
            ni = radial / r_norm if r_norm > 1e-6 else np.zeros(2)

        # Ensure the normal points outward (away from the mesh centroid)
        sign = np.dot(ni, coords[node] - global_centroid)
        if sign < 0:
            ni = -ni

        normal_vecs[i] = ni

    return normal_vecs

def compute_M_matrix(node_idx: int, support_nodes: npt.NDArray[np.int_], coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Computes the M matrix for a given node.

    Parameters
    ----------
    node_idx : int
        Index of the central node.
    support_nodes : npt.NDArray[np.int_]
        Indices of the support nodes.
    coords : npt.NDArray[np.float64]
        Array with shape (n, 2) containing the coordinates of the n nodes.

    Returns
    -------
    npt.NDArray[np.float64]
        The M matrix.
    """
    p0 = coords[node_idx]
    M = np.zeros((6, support_nodes.shape[0]))
    delta_x = coords[support_nodes, 0] - p0[0]
    delta_y = coords[support_nodes, 1] - p0[1]
    M[0, :] = 1
    M[1, :] = delta_x
    M[2, :] = delta_y
    M[3, :] = delta_x ** 2
    M[4, :] = delta_x * delta_y
    M[5, :] = delta_y ** 2
    return M