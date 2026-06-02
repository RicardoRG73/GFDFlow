
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
    coords: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Computes normal vectors at boundary nodes.

    Parameters
    ----------
    boundary_nodes : npt.NDArray[np.int_]
        index of boundary nodes.
    coords : npt.NDArray[np.float64]
        array with shape (n,2) containing the coordinates of the n nodes.

    Returns
    -------
    normal_vecs : npt.NDArray[np.float64]
        array with shape (N,2) containing the normal vectors at the N boundary nodes.
    """
    line_tolerance = 0.99
    N = boundary_nodes.shape[0]
    
    if N < 2:
        return np.zeros((N, 2))

    line_1 = coords[boundary_nodes[1], :] - coords[boundary_nodes[0], :]
    norm_1 = np.linalg.norm(line_1)
    if norm_1 > 0:
        line_1 = line_1 / norm_1
    
    # Check if boundary is a line by comparing first and middle nodes
    line_2 = coords[boundary_nodes[N // 2], :] - coords[boundary_nodes[0], :]
    norm_2 = np.linalg.norm(line_2)
    if norm_2 > 0:
        line_2 = line_2 / norm_2
    
    is_line = np.dot(line_1, line_2) > line_tolerance
    clockwise_rotation = np.array([[0, 1], [-1, 0]])
    
    if is_line:
        line_normal = clockwise_rotation @ line_1
        normal_vecs = np.tile(line_normal, (N, 1))
    else:
        normal_vecs = np.zeros((N, 2))
        centroid = np.mean(coords, axis=0)

        for i, node in enumerate(boundary_nodes):
            distance = np.sqrt((coords[node, 0] - coords[boundary_nodes, 0])**2 + 
                               (coords[node, 1] - coords[boundary_nodes, 1])**2)
            # Use at most 7 closest nodes or N nodes if N < 7
            max_closest = min(7, N)
            closest_nodes = boundary_nodes[distance.argsort()[:max_closest]]
            closest_centroid = np.mean(coords[closest_nodes, :], axis=0)
            
            # Need at least 2 nodes to form vectors for rotation
            if len(closest_nodes) >= 2:
                v1 = coords[closest_nodes[-2]] - closest_centroid
                v2 = coords[closest_nodes[-1]] - closest_centroid
                diff_v = v2 - v1
                norm_diff = np.linalg.norm(diff_v)
                if norm_diff > 0:
                    ni = clockwise_rotation @ diff_v / norm_diff
                    # Ensure normal points outward
                    ni = ni * np.sign(np.dot(ni, coords[node] - centroid))
                    normal_vecs[i] = ni
    
    return normal_vecs
    
def normal_vector_in_node(node_idx: int, boundary_nodes: npt.NDArray[np.int_], coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Computes the normal vector at a given node using boundary nodes around it.

    Parameters
    ----------
    node_idx : int
        Index of the central node.
    boundary_nodes : npt.NDArray[np.int_]
        Indices of the boundary nodes.
    coords : npt.NDArray[np.float64]
        Array with shape (n, 2) containing the coordinates of the n nodes.

    Returns
    -------
    npt.NDArray[np.float64]
        The normal vector at the given node.
    """
    
    tol = 1e-4
    p0 = coords[node_idx]
    p1 = coords[boundary_nodes[0]]
    p2 = coords[boundary_nodes[1]]
    v = p1 - p0
    v = v / np.linalg.norm(v)
    w = p2 - p0
    w = w / np.linalg.norm(w)
    line = np.dot(v, w) > (1 - tol)
    if line:
        clockwise_rotation = np.array([[0, 1], [-1, 0]])
        ni = clockwise_rotation @ v
    else:
        pm = np.mean(coords[boundary_nodes])
        ni = coords[node_idx] - pm
        ni = ni / np.linalg.norm(ni)

    return ni
    
def GfdmTransientStepPicard(p, t, L, f, mat_pr, dir_pr, neu_pr, intrf_pr, intrf_intersc, 
                            u_prev, dt, u0=None, tol=1e-3, max_iter=20, double_nodes=False, is_hydraulic_head=True, omega=0.5):
    """
    Solves one time step for the transient Richards equation.
    
    Parameters
    -----
    u_prev : ndarray
        Solution at the previous time step.
    dt : float
        Time step size.
    u0 : ndarray, optional
        Initial guess for the current time step (if None, uses u_prev).
    """
    from .GFDM import VanGenuchten
    N = p.shape[0]
    if u0 is None:
        u = u_prev.copy()
    else:
        u = u0.copy()

    node_to_idx = {tuple(pos): i for i, pos in enumerate(p)}
    
    # Identify Dirichlet nodes
    dir_nodes = []
    for key in dir_pr:
        dir_nodes.extend(dir_pr[key][0])
    dir_nodes = set(dir_nodes)

    for it in range(max_iter):
        print(f"  Picard Iteration {it+1}/{max_iter}")
        
        curr_mat_pr = {}
        curr_C = np.zeros(N)
        
        for key in mat_pr:
            nodes, k_obj = mat_pr[key]
            if isinstance(k_obj, VanGenuchten):
                # We use a closure to capture current solution 'u' and node mapping
                def mk_k(k_static, u_curr, mapping, is_H):
                    if is_H:
                        return lambda pos: k_static.K(u_curr[mapping[tuple(pos)]] - pos[1])
                    else:
                        return lambda pos: k_static.K(u_curr[mapping[tuple(pos)]])
                curr_mat_pr[key] = [nodes, mk_k(k_obj, u, node_to_idx, is_hydraulic_head)]
                
                # Compute C(h) for these nodes
                for idx in nodes:
                    pos = p[idx]
                    h = u[idx] - pos[1] if is_hydraulic_head else u[idx]
                    curr_C[idx] = k_obj.C(h)
            else:
                curr_mat_pr[key] = [nodes, k_obj]
                
        K_new, F_new = GfdmInterf(p, t, L, f, curr_mat_pr, dir_pr, neu_pr, intrf_pr, intrf_intersc, double_nodes=double_nodes)
        
        # Add transient terms for Non-Dirichlet nodes
        for i in range(N):
            if i not in dir_nodes:
                c_val = curr_C[i]
                K_new[i, i] -= c_val / dt
                F_new[i] -= (c_val / dt) * u_prev[i]
        
        try:
            u_next = np.linalg.solve(K_new, F_new)
        except np.linalg.LinAlgError:
            print("  Singular matrix in Picard iteration. Attempting pseudo-inverse.")
            u_next = np.linalg.pinv(K_new) @ F_new
            
        u_next = omega * u_next + (1 - omega) * u
        # Strictly enforce Dirichlet boundary conditions on the under-relaxed solution
        for key in dir_pr:
            nodes, fd = dir_pr[key]
            for idx in nodes:
                u_next[idx] = fd(p[idx])
            
        err = np.linalg.norm(u_next - u) / (np.linalg.norm(u_next) + 1e-10)
        print(f"  Error: {err:.2e}")
        
        u = u_next.copy()
        
        if err < tol:
            print(f"  Picard converged in {it+1} iterations.")
            return u
            
    print("  Picard did not converge.")
    return u    

