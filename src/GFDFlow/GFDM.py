import numpy as np
import scipy.sparse as sp
from typing import Callable, Dict, List, Tuple, Union, Optional
import numpy.typing as npt
from .utils import get_support_nodes, compute_normal_vectors, normal_vector_in_node

class GFDMI_2D_problem:
    """
    Class to model and solve 2D partial differential equations using the 
    Generalized Finite Difference Method (GFDM) with support for material 
    interfaces and different boundary conditions.

    The solver handles PDEs of the form:
    Au + B*u_x + C*u_y + D*u_xx + E*u_xy + F*u_yy = f

    Attributes
    ----------
    coords : npt.NDArray[np.float64]
        Coordinates of the nodes in the mesh (n_nodes, 2).
    triangles : npt.NDArray[np.int_]
        Connectivity of the nodes forming the mesh triangles (n_triangles, 3).
    L : npt.NDArray[np.float64]
        Coefficients of the differential operator [A, B, C, D, E, F].
    source : Callable[[npt.NDArray[np.float64]], float]
        Function defining the source term f(x, y).
    materials : Dict
        Storage for material properties (permeability) and their associated nodes.
    neumann_boundaries : Dict
        Storage for Neumann boundary conditions.
    dirichlet_boundaries : Dict
        Storage for Dirichlet boundary conditions.
    interfaces : Dict
        Storage for jump conditions and properties at material interfaces.
    """

    def __init__(
        self, 
        coords: npt.NDArray[np.float64], 
        triangles: npt.NDArray[np.int_], 
        L: npt.NDArray[np.float64], 
        source: Callable[[npt.NDArray[np.float64]], float]
    ):
        """
        Initializes the GFDMI 2D problem.

        Parameters
        ----------
        coords : npt.NDArray[np.float64]
            Array with shape (n, 2) containing coordinates of the n nodes.
        triangles : npt.NDArray[np.int_]
            Array with shape (m, 3) containing indices of the m triangles nodes.
        L : npt.NDArray[np.float64]
            Array [A, B, C, D, E, F] of differential operator coefficients.
        source : Callable[[npt.NDArray[np.float64]], float]
            Function taking a point [x, y] and returning the source value.
        
        Raises
        ------
        ValueError
            If input array dimensions are inconsistent.
        """
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must have shape (n, 2), got {coords.shape}")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(f"triangles must have shape (m, 3), got {triangles.shape}")
        if L.size != 6:
            raise ValueError(f"L must have 6 coefficients [A, B, C, D, E, F], got {L.size}")

        self.coords = coords
        self.triangles = triangles
        self.L = L.copy().astype(float)
        self.source = source
        self.materials: Dict[str, List] = {}
        self.neumann_boundaries: Dict[str, List] = {}
        self.dirichlet_boundaries: Dict[str, List] = {}
        self.interfaces: Dict[str, List] = {}
        self.intersections: Dict[str, List] = {}

    def support_nodes(self, node: int, min_support_nodes: int = 5, max_iter: int = 2) -> npt.NDArray[np.int_]:
        """
        Public method to get support nodes for a given node.
        
        Parameters
        ----------
        node : int
            Index of the central node.
        min_support_nodes : int, optional
            Default is 5.
        max_iter : int, optional
            Default is 2.
            
        Returns
        -------
        npt.NDArray[np.int_]
            Indices of the support nodes.
        """
        return get_support_nodes(node, self.triangles, min_support_nodes, max_iter)

    def normal_vector_in_node(self, node: int, boundary_nodes: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
        """
        Public method to get the normal vector at a given node.
        
        Parameters
        ----------
        node : int
            Index of the central node.
        boundary_nodes : npt.NDArray[np.int_]
            Indices of the boundary nodes.
            
        Returns
        -------
        npt.NDArray[np.float64]
            Normal vector at the given node.
        """
        return normal_vector_in_node(node, boundary_nodes, self.coords)
    
    def normal_vectors(self, boundary_nodes: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
        """
        Computes normal vectors at boundary nodes.

        Parameters
        ----------
        boundary_nodes : npt.NDArray[np.int_]
            Indices of boundary nodes.

        Returns
        -------
        npt.NDArray[np.float64]
            Normal vectors (N, 2).
        """
        return compute_normal_vectors(boundary_nodes, self.coords)
    
    def material(self, label: str, permeability: Callable[[npt.NDArray[np.float64]], float], interior_nodes: npt.NDArray[np.int_]) -> None:
        """Defines material properties for a set of nodes."""
        self.materials[label] = [permeability, interior_nodes]

    def neumann_boundary(self, label: str, permeability: Callable[[npt.NDArray[np.float64]], float], 
                         boundary_nodes: npt.NDArray[np.int_], condition: Callable[[npt.NDArray[np.float64]], float]) -> None:
        """Defines Neumann boundary conditions."""
        self.neumann_boundaries[label] = [permeability, boundary_nodes, condition]

    def dirichlet_boundary(self, label: str, boundary_nodes: npt.NDArray[np.int_], 
                           condition: Callable[[npt.NDArray[np.float64]], float]) -> None:
        """Defines Dirichlet boundary conditions."""
        self.dirichlet_boundaries[label] = [boundary_nodes, condition]

    def interface(
            self,
            label: str,
            k_left: Callable[[npt.NDArray[np.float64]], float],
            k_right: Callable[[npt.NDArray[np.float64]], float],
            nodes_left: npt.NDArray[np.int_],
            nodes_right: npt.NDArray[np.int_],
            beta: Callable[[npt.NDArray[np.float64]], float],
            alpha: Callable[[npt.NDArray[np.float64]], float],
            interior_left: npt.NDArray[np.int_],
            interior_right: npt.NDArray[np.int_]
        ) -> None:
        """Defines interface conditions between two materials."""
        self.interfaces[label] = [
            k_left, k_right, nodes_left, nodes_right, beta, alpha, interior_left, interior_right
        ]

    def intersection(
        self,
        label: str,
        intersection_node: int,
        interface_left: str,
        interface_right: str,
        material_between: str,
        f_intersection: Callable[[npt.NDArray[np.float64]], float]
    ) -> None:
        """Defines intersection between two interfaces."""
        self.intersections[label] = [
            intersection_node, interface_left, interface_right, material_between, f_intersection
        ]

    def _assemble_point_discretization(self, i: int, k_val: float, operator: npt.NDArray[np.float64], 
                                       support_indices: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
        """Calculates the Gamma stencil for a node."""
        deltasx = self.coords[support_indices, 0] - self.coords[i, 0]
        deltasy = self.coords[support_indices, 1] - self.coords[i, 1]
        
        # Taylor expansion matrix (size 6 x support_size)
        M = np.vstack((
            np.ones(deltasx.shape),
            deltasx,
            deltasy,
            deltasx**2,
            deltasx*deltasy,
            deltasy**2
        ))
        
        # solve M^T * Gamma = k * L
        return np.linalg.pinv(M) @ (k_val * operator)

    def discretization_K_F(self, continuous: bool = False) -> Tuple[sp.csr_matrix, npt.NDArray[np.float64]]:
        """
        Assembles the global stiffness matrix K and force vector F.
        
        Parameters
        ----------
        continuous : bool, optional
            If True, uses the continuous discretization variant. Default is False.
            
        Returns
        -------
        K : sp.csr_matrix
            Global matrix (N, N).
        F : npt.NDArray[np.float64]
            Global vector (N,).
        """
        N = self.coords.shape[0]
        K = sp.lil_matrix((N, N))
        F = np.zeros(N)
        
        # Adjust scaling for second order terms (Taylor expansion matches [1, x, y, x^2/2, xy, y^2/2])
        # But here the input L is [A, B, C, D, E, F] matching [1, x, y, x^2, xy, y^2]?
        # Looking at original code: L[3] *= 2 and L[5] *= 2.
        # This means D and F correspond to x^2 and y^2 which should be divided by 2 in stencil calc if L is original.
        # Actually, if we multiply L[3] and L[5] by 2, we are compensating for the x^2/2 and y^2/2 in Taylor.
        L = self.L
        L[3] *= 2
        L[5] *= 2

        # 1. Identify interior nodes (exclude those in Neumann boundaries)
        neumann_n = np.array([], dtype=int)
        for val in self.neumann_boundaries.values():
            neumann_n = np.union1d(neumann_n, val[1])

        # 2. Interior nodes assembly
        for k_fn, nodes in self.materials.values():
            interior = np.setdiff1d(nodes, neumann_n)
            for i in interior:
                I = self.support_nodes(i)
                Gamma = self._assemble_point_discretization(i, k_fn(self.coords[i]), L, I)
                K[i, I] = Gamma
                F[i] = self.source(self.coords[i])

        # 3. Neumann boundary assembly
        for k_fn, b_nodes, u_n_fn in self.neumann_boundaries.values():
            normals = self.normal_vectors(b_nodes)
            for idx, i in enumerate(b_nodes):
                I = self.support_nodes(i)
                ni = normals[idx]
                k_val = k_fn(self.coords[i])
                
                # Setup ghost point logic
                deltasx = self.coords[I, 0] - self.coords[i, 0]
                deltasy = self.coords[I, 1] - self.coords[i, 1]
                
                # Estimate distance for ghost point
                mean_h = np.mean(np.sqrt(deltasx[1:]**2 + deltasy[1:]**2)) if len(I) > 1 else 0.1
                ghost_x, ghost_y = ni * mean_h
                
                # Augmented stencil
                aug_dx = np.insert(deltasx, 0, ghost_x)
                aug_dy = np.insert(deltasy, 0, ghost_y)
                
                M_aug = np.vstack((np.ones(aug_dx.shape), aug_dx, aug_dy, aug_dx**2, aug_dx*aug_dy, aug_dy**2))
                M_pinv = np.linalg.pinv(M_aug)
                
                Gamma_full = M_pinv @ (k_val * L)
                Gamma_ghost = Gamma_full[0]
                Gamma_nodes = Gamma_full[1:]
                
                # Normal derivative operator [0, nx, ny, 0, 0, 0]
                Normal_op = np.array([0, ni[0], ni[1], 0, 0, 0])
                Gamma_n_full = M_pinv @ (k_val * Normal_op)
                Gamma_n_ghost = Gamma_n_full[0]
                Gamma_n_nodes = Gamma_n_full[1:]
                
                Gg = Gamma_ghost / Gamma_n_ghost
                K[i, I] = Gamma_nodes - Gg * Gamma_n_nodes
                F[i] = self.source(self.coords[i]) - Gg * u_n_fn(self.coords[i])

        # 4. Interface assembly
        for k0, k1, biA, biB, beta, alpha, m0, m1 in self.interfaces.values():
            if not continuous:
                # Discontinuous interface handling
                normals = self.normal_vectors(biA)
                for idx, i in enumerate(biA):
                    I0 = np.setdiff1d(self.support_nodes(i), m1)
                    ni = normals[idx]
                    k0_val = k0(self.coords[i])
                    
                    # Similar ghost logic as Neumann
                    deltasx = self.coords[I0, 0] - self.coords[i, 0]
                    deltasy = self.coords[I0, 1] - self.coords[i, 1]
                    mean_h = np.mean(np.sqrt(deltasx[1:]**2 + deltasy[1:]**2)) if len(I0) > 1 else 0.1
                    gx, gy = ni * mean_h
                    
                    aug_dx = np.insert(deltasx, 0, gx)
                    aug_dy = np.insert(deltasy, 0, gy)
                    M_aug = np.vstack((np.ones(aug_dx.shape), aug_dx, aug_dy, aug_dx**2, aug_dx*aug_dy, aug_dy**2))
                    M_pinv = np.linalg.pinv(M_aug)
                    
                    G_full = M_pinv @ (k0_val * L)
                    Gn_full = M_pinv @ (k0_val * np.array([0, ni[0], ni[1], 0, 0, 0]))
                    
                    Gg = G_full[0] / Gn_full[0]
                    K[i, I0] = G_full[1:] - Gg * Gn_full[1:]
                    F[i] = self.source(self.coords[i]) - Gg * beta(self.coords[i])
                
                normals_B = self.normal_vectors(biB)
                for idx, i in enumerate(biB):
                    I1 = np.setdiff1d(self.support_nodes(i), m0)
                    ni = -normals_B[idx]
                    k1_val = k1(self.coords[i])
                    
                    deltasx = self.coords[I1, 0] - self.coords[i, 0]
                    deltasy = self.coords[I1, 1] - self.coords[i, 1]
                    mean_h = np.mean(np.sqrt(deltasx[1:]**2 + deltasy[1:]**2)) if len(I1) > 1 else 0.1
                    gx, gy = ni * mean_h
                    
                    aug_dx = np.insert(deltasx, 0, gx)
                    aug_dy = np.insert(deltasy, 0, gy)
                    M_aug = np.vstack((np.ones(aug_dx.shape), aug_dx, aug_dy, aug_dx**2, aug_dx*aug_dy, aug_dy**2))
                    M_pinv = np.linalg.pinv(M_aug)
                    
                    G_full = M_pinv @ (k1_val * L)
                    Gn_full = M_pinv @ (k1_val * np.array([0, ni[0], ni[1], 0, 0, 0]))
                    
                    Gg = G_full[0] / Gn_full[0]
                    
                    # Link to counterpart on side A
                    dist = np.linalg.norm(self.coords[biA] - self.coords[i], axis=1)
                    biA_idx = biA[np.argmin(dist)]
                    
                    # Contribution to Side A interface node
                    # Note: Original code adds this via K[biA_i, I0] += ... which seems to imply re-discretizing or just adding components
                    # I'll stick to the original logic pattern for correctness
                    K[biA_idx, I1] += G_full[1:] - Gg * Gn_full[1:]
                    F[biA_idx] += self.source(self.coords[i]) - Gg * beta(self.coords[i])
                    
                    # Side B interface node just takes jump condition
                    K[i, biA_idx] = -1
                    K[i, i] = 1
                    F[i] = alpha(self.coords[i])
            else:
                # Continuous interface assembly
                # Side A
                normals = self.normal_vectors(biA)
                for idx, i in enumerate(biA):
                    I0 = np.setdiff1d(self.support_nodes(i), m1)
                    ni = normals[idx]
                    k0_val = k0(self.coords[i])
                    
                    deltasx = self.coords[I0, 0] - self.coords[i, 0]
                    deltasy = self.coords[I0, 1] - self.coords[i, 1]
                    mean_h = np.mean(np.sqrt(deltasx[1:]**2 + deltasy[1:]**2)) if len(I0) > 1 else 0.1
                    gx, gy = ni * mean_h
                    
                    aug_dx = np.insert(deltasx, 0, gx)
                    aug_dy = np.insert(deltasy, 0, gy)
                    M_aug = np.vstack((np.ones(aug_dx.shape), aug_dx, aug_dy, aug_dx**2, aug_dx*aug_dy, aug_dy**2))
                    M_pinv = np.linalg.pinv(M_aug)
                    
                    G_full = M_pinv @ (k0_val * L)
                    Gn_full = M_pinv @ (k0_val * np.array([0, ni[0], ni[1], 0, 0, 0]))
                    
                    Gg = G_full[0] / Gn_full[0]
                    K[i, I0] = G_full[1:] - Gg * Gn_full[1:]
                    F[i] = self.source(self.coords[i]) - Gg * beta(self.coords[i])
                
                # Side B (Continuous version adds to the same interface node)
                for idx, i in enumerate(biA): # biA here as same nodes but for Side B contribution
                    I1 = np.setdiff1d(self.support_nodes(i), m0)
                    ni = -normals[idx]
                    k1_val = k1(self.coords[i])
                    
                    deltasx = self.coords[I1, 0] - self.coords[i, 0]
                    deltasy = self.coords[I1, 1] - self.coords[i, 1]
                    mean_h = np.mean(np.sqrt(deltasx[1:]**2 + deltasy[1:]**2)) if len(I1) > 1 else 0.1
                    gx, gy = ni * mean_h
                    
                    aug_dx = np.insert(deltasx, 0, gx)
                    aug_dy = np.insert(deltasy, 0, gy)
                    M_aug = np.vstack((np.ones(aug_dx.shape), aug_dx, aug_dy, aug_dx**2, aug_dx*aug_dy, aug_dy**2))
                    M_pinv = np.linalg.pinv(M_aug)
                    
                    G_full = M_pinv @ (k1_val * L)
                    Gn_full = M_pinv @ (k1_val * np.array([0, ni[0], ni[1], 0, 0, 0]))
                    
                    Gg = G_full[0] / Gn_full[0]
                    K[i, I1] = K[i,I1].toarray() + (G_full[1:] - Gg * Gn_full[1:])
                    F[i] += self.source(self.coords[i]) - Gg * beta(self.coords[i])

        # 5. Interface intersection assembly
        for center_node, b_left, b_right, mat_between, beta_center in self.intersections.values():
            if continuous:
                k = self.materials[mat_between][1]
                # interface nodes
                left_nodes = self.interfaces[b_left][2]
                right_nodes = self.interfaces[b_right][2]
                l_r_nodes = np.union1d(left_nodes, right_nodes)
                # material nodes
                mat_nodes = self.materials[mat_between][1]     # nodes
                k_func =  self.materials[mat_between][0]  # source
                ki = k_func(self.coords[center_node])
                
                # support nodes and normal vec at center_node
                all_nodes = np.union1d(l_r_nodes, mat_nodes)
                all_nodes = np.union1d(all_nodes, [center_node])
                support_nodes = self.support_nodes(center_node)
                support_nodes = np.intersect1d(support_nodes, all_nodes)
                n_center = self.normal_vector_in_node(center_node, l_r_nodes)
                # GFDM weights
                deltasx = self.coords[support_nodes, 0] - self.coords[center_node, 0]
                deltasy = self.coords[support_nodes, 1] - self.coords[center_node, 1]
                mean_h = np.mean(np.sqrt(deltasx[1:]**2 + deltasy[1:]**2)) if len(support_nodes) > 1 else 0.1
                gx, gy = n_center * mean_h
                
                aug_dx = np.insert(deltasx, 0, gx)
                aug_dy = np.insert(deltasy, 0, gy)
                M_aug = np.vstack((np.ones(aug_dx.shape), aug_dx, aug_dy, aug_dx**2, aug_dx*aug_dy, aug_dy**2))
                M_pinv = np.linalg.pinv(M_aug)
                
                G_full = M_pinv @ (ki * L)
                Gn_full = M_pinv @ (ki * np.array([0, n_center[0], n_center[1], 0, 0, 0]))
                
                Gg = G_full[0] / Gn_full[0]
                K[center_node, support_nodes] = K[center_node, support_nodes].toarray() + (G_full[1:] - Gg * Gn_full[1:])
                F[center_node] += self.source(self.coords[center_node]) - Gg * beta_center(self.coords[center_node])
                
            else:
                pass
            
        # 6. Dirichlet boundary assembly
        for b_nodes, u_fn in self.dirichlet_boundaries.values():
            for i in b_nodes:
                K[i, :] = 0
                K[i, i] = 1
                F[i] = u_fn(self.coords[i])

        return K.tocsr(), F

    def discontinuous_discretization(self) -> Tuple[sp.csr_matrix, npt.NDArray[np.float64]]:
        """Backwards compatibility for discontinuous assembly."""
        return self.discretization_K_F(continuous=False)

    def continuous_discretization(self) -> Tuple[sp.csr_matrix, npt.NDArray[np.float64]]:
        """Backwards compatibility for continuous assembly."""
        return self.discretization_K_F(continuous=True)
