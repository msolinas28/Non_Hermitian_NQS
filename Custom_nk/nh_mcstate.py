from netket.vqs import MCState
from cc_nh_expect import expect as expect_symm_cc
from nh_expect import expect

# Register custom expectation functions in the dictionary to enable new symmetries.
# Each custom function must follow the same signature as `cc_nh_expect`.
symmetries = {"cc": expect_symm_cc}

class NHMCState(MCState):
    """
    Variational Monte Carlo (VMC) state class for non-Hermitian systems with optional symmetry.

    This class extends NetKet's :class:`MCState` to support:
    - Biorthogonal states via `bstate` (left or right),
    - Symmetry-specific expectation values via a registry of custom functions,
    - Non-Hermitian expectation computations between two variational states.

    Parameters
    ----------
    bstate : str
        Specifies whether this state is the "left" or "right" state in a biorthogonal pair.
    symmetry : str, optional
        Name of the symmetry to use for computing expectation values. Must be registered
        in the `symmetries` dictionary. Default is `None`.
    **kwargs
        Additional keyword arguments passed to the base :class:`MCState`.
    """
    def __init__(self, bstate, symmetry=None, **kwards):
        super().__init__(**kwards)
        self.bstate = bstate
        self.symmetry = symmetry
    
    @property
    def bstate(self):
        return self._bstate

    @bstate.setter
    def bstate(self, val):
        if val.lower() not in ["left", "right"]:
            raise ValueError(f"The variable bstate can only be 'Left' or 'Right', got {val}")
        self._bstate = val

    @property
    def symmetry(self):
        return self._symmetry
    
    @symmetry.setter
    def symmetry(self, val):
        if val is not None: 
            if val.lower() not in symmetries.keys():
                raise ValueError(
                    f"Invalid symmetry '{val}'. "
                    f"Allowed values are: {', '.join(symmetries.keys())}"
                )
        self._symmetry = val

    def nh_expect_symm(self, O, jax_operator=False):
        """
        Compute the biorthogonal expectation value of an operator using the
        symmetry-specific method registered in `symmetries`.

        Parameters
        ----------
        O : AbstractOperator
            Operator whose expectation value is computed.
        jax_operator : bool, optional
            Whether O is a JAX-operator or not (default: False).

        Returns
        -------
            An estimation of the quantum expectation value \frac{\langle\tilde\psi | O | \psi\rangle}{\langle\tilde\psi|\psi\rangle}.
        """
        if self.symmetry is not None:  
            return symmetries[self.symmetry.lower()](
                O,
                self.samples,
                self.model.apply,
                self.variables,
                jax_operator,
                self.chunk_size
            )
        else: 
            raise ValueError(f"The property symmetry has to be defined, got None")
    
    def nh_expect(self, O, vs_tilde, PDF, jax_operator=False):
        """
        Compute the biorthogonal expectation value of an operator using this state and
        another variational state (`vs_tilde`).

        Dispatches automatically to left or right evaluation based on `bstate`.

        Parameters
        ----------
        O : AbstractOperator
            Operator to evaluate.
        vs_tilde : NHMCState
            The partner variational state (left or right).
        PDF : str
            Probability distribution to sample from ("left" or "right").
        jax_operator : bool, optional
            Whether O is a JAX-operator or not (default: False).

        Returns
        -------
        complex or AbstractOperator
            An estimation of the quantum expectation value \frac{\langle\tilde\psi | O | \psi\rangle}{\langle\tilde\psi|\psi\rangle}.
        """
        if self.bstate.lower() == "left":
            return self._nh_expect_l(O, vs_tilde, PDF, jax_operator)
        else:
            return self._nh_expect_r(O, vs_tilde, PDF, jax_operator)
    
    def _nh_expect_r(self, O, vs_l, PDF, jax_operator):
        exp_fun = lambda x: expect(
            O,
            x,
            vs_l.model.apply, 
            vs_l.variables,
            self.model.apply,
            self.variables,
            PDF,
            jax_operator,
            self.chunk_size
        )

        if PDF.lower() == "left":
            return exp_fun(vs_l.samples)
        elif PDF.lower() == "right":
            return exp_fun(self.samples)
        else:
            raise ValueError("PDF has to be either 'Left' or 'Right'.")
        
    def _nh_expect_l(self, O, vs_r, PDF, jax_operator):
        exp_fun = lambda x: expect(
            O,
            x,
            self.model.apply,
            self.variables,
            vs_r.model.apply,
            vs_r.variables,
            PDF,
            jax_operator,
            self.chunk_size
        )

        if PDF.lower() == "left":
            return exp_fun(self.samples)
        elif PDF.lower() == "right":
            return exp_fun(vs_r.samples)
        else:
            raise ValueError("PDF has to be either 'Left' or 'Right'.")