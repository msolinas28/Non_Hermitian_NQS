from netket.operator.spin import sigmax, sigmaz
import netket as nk
from functools import cached_property
import numpy as np
import jax.numpy as jnp

class cTFIM():
    """
    Complex Transverse-Field Ising Model.

    This class defines a non-Hermitian extension of the transverse-field Ising model
    on a given graph. The Hamiltonian includes:
      - Ising interactions along the z-axis,
      - A transverse field along the x-axis,
      - A purely imaginary longitudinal field along the z-axis, 
        which makes the model non-Hermitian.

    The Hamiltonian takes the form

        H = - λ Σ_{(i,j)∈E} σᶻᵢ σᶻⱼ - h Σᵢ σˣᵢ - i k Σᵢ σᶻᵢ ,

    where:
      - λ controls the strength of the Ising coupling,
      - h controls the transverse field strength,
      - k controls the non-Hermitian longitudinal field strength,
      - E is the set of edges of the graph.

    Parameters
    ----------
    hi : netket.hilbert.AbstractHilbert
        The Hilbert space on which the operator acts (e.g. `nk.hilbert.Spin`).
    graph : netket.graph.AbstractGraph
        The underlying graph defining the lattice connectivity.
    lam : float
        Coupling constant for the Ising interaction (λ).
    h : float
        Strength of the transverse field along the x direction.
    k : float
        Strength of the imaginary longitudinal field along the z direction.
    """
    def __init__(self, hi, graph, lam: float, h: float, k: float):
        self.hi = hi
        self.lam = lam 
        self.h = h
        self.k = k
        self.graph = graph

    @cached_property 
    def operator(self):
        """
        The non-Hermitian Hamiltonian operator H.
        """
        H = nk.operator.LocalOperator(self.hi, dtype = jnp.complex128)

        for (i, j) in self.graph.edges():
            H += - self.lam * sigmaz(self.hi, i) @ sigmaz(self.hi, j)

        for i in range(self.graph.n_nodes):
            H += - self.h * sigmax(self.hi, i) - 1j * self.k * sigmaz(self.hi, i)

        return H
    
    @cached_property 
    def adjoint_operator(self):
        """
        The Hermitian conjugate of H (i.e. H†).
        """
        H = nk.operator.LocalOperator(self.hi, dtype = jnp.complex128)

        for (i, j) in self.graph.edges():
            H += - self.lam * sigmaz(self.hi, i) @ sigmaz(self.hi, j)

        for i in range(self.graph.n_nodes):
            H += - self.h * sigmax(self.hi, i) + 1j * self.k * sigmaz(self.hi, i)

        return H
    
    @property
    def real_lower_bound(self):
        """
        Lower bound for the real part of the spectrum.
        """
        return - self.graph.n_nodes * np.abs(self.h) - self.graph.n_edges * np.abs(self.lam)
    
    @property
    def imag_lower_bound(self):
        """
        Lower bound for the imaginary part of the spectrum.
        """
        return - self.graph.n_nodes * np.abs(self.k)
    
    @property
    def lower_bound(self):
        """
        Combined complex lower bound.
        """
        E_r = self.real_lower_bound
        E_i = self.imag_lower_bound

        return E_r + 1j * E_i