import jax
import jax.numpy as jnp
import netket as nk

class Variance:
    """
    Variance loss operator for non-Hermitian quantum systems.

    This class constructs the operator corresponding to the generalized variance
    functional used as a cost function in variational Monte Carlo (VMC) for
    biorthogonal states. It takes as input a Hamiltonian (O) and its adjoint (O_adj),
    and produces two operators:
    
    - the left-variance operator (acting on the left state),
    - the right-variance operator (acting on the right state).

    The variance operator depends on a reference energy ε, which must be updated
    dynamically during optimization.

    Mathematically, the variance operator is defined as

    .. math::

        V_{\text{left}}(ε) &= (O - ε)\,(O^† - ε^*) \\
        V_{\text{right}}(ε) &= (O^† - ε^*)\,(O - ε)

    where
    - O is the Hamiltonian,
    - O^† is its Hermitian conjugate
    - ε is the current estimate of the energy expectation value.

    Parameters
    ----------
    O : AbstractOperator
        The Hamiltonian operator.
    O_adj : AbstractOperator
        The adjoint (Hermitian conjugate) of the Hamiltonian.
    """
    def __init__(self, O, O_adj):
        OO_l = O @ O_adj
        OO_r = O_adj @ O
        Id = nk.operator.spin.identity(O.hilbert)

        self._update_l = lambda x: self._var_fun(x, OO=OO_l, O=O, O_adj=O_adj, Id=Id)
        self._update_r = lambda x: self._var_fun(x, OO=OO_r, O=O, O_adj=O_adj, Id=Id)
        self._right = None
        self._left = None

    @staticmethod
    def _var_fun(eps, *, OO, O, O_adj, Id):
        eps_abs = jnp.pow(jnp.abs(eps), 2)
        eps_conj = jax.numpy.conj(eps)
        return OO + (Id * eps_abs) - (O * eps_conj) - (O_adj * eps)

    @property
    def left(self):
        """
        Returns the left variance operator.

        Returns
        -------
        AbstractOperator
            The left variance operator.
        """
        if self._left is None:
            raise AttributeError("Run .update(eps) before accessing this property.")
        return self._left

    @property
    def right(self):
        """
        Returns the right variance operator.

        Returns
        -------
        AbstractOperator
            The right variance operator.
        """
        if self._right is None:
            raise AttributeError("Run .update(eps) before accessing this property.")
        return self._right

    def update_l(self, eps):
        """
        Updates the left variance operator with the given ε.

        Parameters
        ----------
        eps : complex
            The current estimate of the energy ε.
        """
        self._left = self._update_l(eps)

    def update_r(self, eps):
        """
        Updates the right variance operator with the given ε.

        Parameters
        ----------
        eps : complex
            The current estimate of the energy ε.
        """
        self._right = self._update_r(eps)
    
    def update(self, eps):
        """
        Updates both left and right variance operators with the given ε.

        Parameters
        ----------
        eps : complex
            The current estimate of the energy ε.
        """
        self.update_l(eps)
        self.update_r(eps)