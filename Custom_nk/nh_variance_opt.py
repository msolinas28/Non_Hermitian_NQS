from netket.utils import timing
from netket.jax import tree_cast
from netket.driver import VMC
from collections.abc import Callable, Iterable
import numbers
from functools import partial
from netket import config
import copy
import optax
import jax
from tqdm.auto import tqdm
from netket.logging import AbstractLog, JsonLog
from netket.operator._abstract_observable import AbstractObservable
from netket.utils import mpi, timing
from netket.driver import AbstractVariationalDriver, VMC
from netket.driver.abstract_variational_driver import _to_iterable
CallbackT = Callable[[int, dict, "AbstractVariationalDriver"], bool]
from netket.optimizer import identity_preconditioner, PreconditionerT
from netket.stats import Stats
import math

from variance import Variance
from nh_mcstate import NHMCState

def _format_stats(stats: Stats):
    mean = f"{stats.mean.real:.3e}"
    err = f"{stats.error_of_mean:.3e}"
    var  = f"{stats.variance:.3e}"

    if not math.isnan(stats.R_hat):
        ext = f", R̂={stats.R_hat:.4f}"
    else:
        ext = ""
    if config.netket_experimental_fft_autocorrelation:
        if not (math.isnan(stats.tau_corr) and math.isnan(stats.tau_corr_max)):
            ext += f", τ={stats.tau_corr:.1f}<{stats.tau_corr_max:.1f}"
    return f"{mean} ± {err} [σ²={var}{ext}]"

@partial(jax.jit, static_argnums=0)
def apply_gradient(optimizer_fun, optimizer_state, dp, params):

    updates, new_optimizer_state = optimizer_fun(dp, optimizer_state, params)

    new_params = optax.apply_updates(params, updates)

    if config.netket_experimental_sharding:
        sharding = jax.sharding.PositionalSharding(jax.devices()).replicate()
        new_optimizer_state = jax.lax.with_sharding_constraint(
            new_optimizer_state, sharding
        )
        new_params = jax.lax.with_sharding_constraint(new_params, sharding)

    return new_optimizer_state, new_params

class NHDriverSymm(VMC):
    """
    A custom variational Monte Carlo (VMC) driver for non-Hermitian quantum systems 
    with symmetry constraints.

    This driver extends :class:`VMC` to handle optimization of non-Hermitian 
    Hamiltonians in a biorthogonal framework. It introduces additional logic 
    for managing different optimization phases (fixed, transition, and 
    self-consistent) using a dynamically updated reference energy `eps`.

    The cost function optimized by this driver is the *variance* of the Hamiltonian,
    generalized to the non-Hermitian case.

    Parameters
    ----------
    variational_state : NHMCState
        The variational state to be optimized. Must be an instance of 
        :class:`NHMCState` and must have a non-None symmetry defined.
    hamiltonian_adj : AbstractOperator
        The adjoint (Hermitian conjugate) of the Hamiltonian to be optimized.
        It is internally converted to a JAX operator.
    **kwargs
        Additional keyword arguments passed to the base :class:`VMC` driver.
    """
    def __init__(self, variational_state: NHMCState, hamiltonian_adj, **kwargs):
        if not isinstance(variational_state, NHMCState):
            raise TypeError(
                f"Expected variational_state to be NHMCState, got {type(variational_state).__name__}"
            )
        if variational_state.symmetry is None:
            raise ValueError("NHDriverSymm: `variational_state.symmetry` must be set (got None).")

        super().__init__(variational_state=variational_state, **kwargs)
        if self.state.bstate.lower() != "right":
            raise NotImplementedError("The driver only works with variational_state with bstate set to 'Right'.")

        self._ham = self._ham.to_jax_operator()
        self._ham_adj = hamiltonian_adj.to_jax_operator()
        self._loss_name = "Variance"
        self.variance = Variance(self._ham, self._ham_adj)
        self._eps = None

    @property
    def eps(self):
        """
        Returns the current estimate of the reference energy ε.

        Returns
        -------
        complex
            The current value of ε used to update the variance loss.
        """
        return self._eps
        
    def _get_eps(self):
        return self.state.nh_expect_symm(self._ham)
    
    def _trans_step(self):
        alpha = (self._n_fixed + self._n_trans - self._step_count) / self._n_trans 
        self._eps = alpha * self._eps_init + (1 - alpha) * self._get_eps().mean
        self.variance.update_r(self._eps)
    
    def _self_consistent_step(self):
        self._eps = self._get_eps().mean
        self.variance.update_r(self._eps)

    @timing.timed
    def _forward_and_backward(self):

        self.state.reset()
        self.state.sample()
        
        if self.step_count < self._n_fixed:
            self._eps = self._eps_init
            self._phase = "Fixed"
        elif self.step_count < self._n_trans + self._n_fixed:
            self._trans_step()
            self._phase = "Transition"
        else:
            self._self_consistent_step()
            self._phase = "Self-consistent"
        
        self._loss_stats, self._loss_grad = self.state.expect_and_grad(self.variance.right)

        # if it's the identity it does
        # self._dp = self._loss_grad
        self._dp = self.preconditioner(self.state, self._loss_grad, self.step_count)

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp = tree_cast(self._dp, self.state.parameters)

        return self._dp
    
    def run(
        self,
        n_iter: int,
        eps_init: complex,
        fixed_steps_ratio: float,
        transition_steps_ratio: float,
        out: AbstractLog | Iterable[AbstractLog] | str | None = (),
        obs: dict[str, AbstractObservable] | None = None,
        step_size: int = 1,
        show_progress: bool = True,
        save_params_every: int = 50,  # for default logger
        write_every: int = 50,  # for default logger
        callback: CallbackT | Iterable[CallbackT] = lambda *x: True,
        timeit: bool = False,
    ):
        """
        Runs this variational driver, updating the weights of the network stored in
        this driver for `n_iter` steps and dumping values of the observables `obs`
        in the output `logger`.

        It is possible to control more specifically what quantities are logged, when to
        stop the optimisation, or to execute arbitrary code at every step by specifying
        one or more callbacks, which are passed as a list of functions to the keyword
        argument `callback`.

        Callbacks are functions that follow this signature:

        .. Code::

            def callback(step, log_data, driver) -> bool:
                ...
                return True/False

        If a callback returns True, the optimisation continues, otherwise it is stopped.
        The `log_data` is a dictionary that can be modified in-place to change what is
        logged at every step. For example, this can be used to log additional quantities
        such as the acceptance rate of a sampler.

        Loggers are specified as an iterable passed to the keyword argument `out`. If only
        a string is specified, this will create by default a :class:`nk.logging.JsonLog`.
        To know about the output format check its documentation. The logger object is
        also returned at the end of this function so that you can inspect the results
        without reading the json output.

        When running among multiple MPI ranks/Jax devices, the logging logic is executed
        on all nodes, but only root-rank loggers should write to files or do expensive I/O
        operations.

        .. note::

            Before NetKet 3.15, loggers where automatically 'ignored' on non-root ranks.
            However, starting with NetKet 3.15 it is the responsability of a logger to
            check if it is executing on a non-root rank, and to 'do nothing' if that is
            the case.

            The change was required to work correctly and efficiently with sharding. It will
            only affect users that were defining custom loggers themselves.

        Args:
            n_iter: the total number of iterations to be performed during this run.
            eps_init: initial estiamte of epsilon.
            fixed_steps_ratio: fraction of total iterations to keep ε fixed to eps_init
            transition_steps_ratio: Fraction of total iterations used for interpolating ε between eps_init 
                and the current expectation value.
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
                If this argument is a string, it will be used as output prefix for the standard JSON logger.
            obs: An iterable containing all observables that should be computed
            step_size: Every how many steps should observables be logged to disk (default=1)
            callback: Callable or list of callable callback functions to stop training given a condition
            show_progress: If true displays a progress bar (default=True)
            save_params_every: Every how many steps the parameters of the network should be
                serialized to disk (ignored if logger is provided)
            write_every: Every how many steps the json data should be flushed to disk (ignored if
                logger is provided)
            timeit: If True, provide timing information.
        """

        if not isinstance(n_iter, numbers.Number):
            raise ValueError(
                "n_iter, the first positional argument to `run`, must be a number!"
            )

        self._n_iter = n_iter
        self._n_fixed = int(n_iter * fixed_steps_ratio)
        self._n_trans = int(n_iter * transition_steps_ratio) 
        self._eps_init = self._eps = eps_init
        self.variance.update_r(self._eps)

        if obs is None:
            obs = {}

        # if out is a path, create an overwriting Json Log for output
        if isinstance(out, str):
            out = JsonLog(out, "w", save_params_every, write_every)
        elif out is None:
            out = ()

        loggers = _to_iterable(out)
        callbacks = _to_iterable(callback)
        callback_stop = False

        with timing.timed_scope(force=timeit) as timer:
            with tqdm(
                total=n_iter,
                disable=not show_progress or not self._is_root,
                dynamic_ncols=True,
            ) as pbar:
                old_step = self.step_count
                first_step = True

                for step in self.iter(n_iter, step_size):
                    log_data = self.estimate(obs)
                    self._log_additional_data(log_data, step)

                    # if the cost-function is defined then report it in the progress bar
                    if self._loss_stats is not None:
                        pbar.set_postfix_str(
                            f"{self._phase} phase | "
                            f"{self._loss_name} = {_format_stats(self._loss_stats)}"
                        )
                        log_data[self._loss_name] = self._loss_stats
                        log_data["Eps"] = self._eps

                    # Execute callbacks before loggers because they can append to log_data
                    for callback in callbacks:
                        if not callback(step, log_data, self):
                            callback_stop = True

                    with timing.timed_scope(name="loggers"):
                        for logger in loggers:
                            logger(self.step_count, log_data, self.state)

                    if len(callbacks) > 0:
                        if mpi.mpi_any(callback_stop):
                            break

                    # Reset the timing of tqdm after the first step, to ignore compilation time
                    if first_step:
                        first_step = False
                        pbar.unpause()

                    # Update the progress bar
                    pbar.update(self.step_count - old_step)
                    old_step = self.step_count

                # Final update so that it shows up filled.
                pbar.update(self.step_count - old_step)

        # flush at the end of the evolution so that final values are saved to
        # file
        for logger in loggers:
            logger.flush(self.state)

        if timeit:
            self._timer = timer
            if self._is_root:
                print(timer)

        return loggers
    
class NHDriver(VMC):
    """
    A custom variational Monte Carlo (VMC) driver for biorthogonal non-Hermitian 
    quantum systems.

    This driver extends :class:`VMC` to simultaneously optimize two variational 
    states forming a biorthogonal pair: a 'right' state and a 'left' state (denoted
    as `state` and `state_tilde`). It supports sampling from either distribution
    (left or right PDF) and computes gradients for both states using a generalized
    variance cost function.

    Parameters
    ----------
    variational_state : NHMCState
        The 'right' variational state to be optimized. Must be an instance of
        :class:`NHMCState`.
    variational_state_tilde : NHMCState
        The 'left' variational state, also optimized simultaneously. Must be
        an instance of :class:`NHMCState`.
    hamiltonian_adj : AbstractOperator
        The adjoint (Hermitian conjugate) of the Hamiltonian. Converted internally
        to a JAX operator.
    PDF : str, optional
        The probability distribution to sample from, either ``"left"`` or ``"right"``.
        Default is ``"left"``.
    **kwargs
        Additional keyword arguments passed to the base :class:`VMC` driver.
    """
    def __init__(self, variational_state: NHMCState, variational_state_tilde: NHMCState, 
                 hamiltonian_adj, PDF="left", **kwargs):
        if not isinstance(variational_state, NHMCState):
            raise TypeError(
                f"Expected variational_state to be NHMCState, got {type(variational_state).__name__}"
            )
        if not isinstance(variational_state_tilde, NHMCState):
            raise TypeError(
                f"Expected variational_state_tilde to be NHMCState, got {type(variational_state_tilde).__name__}"
            )

        super().__init__(variational_state=variational_state, **kwargs)
        
        self._variational_state_tilde = variational_state_tilde
        if self.state.bstate.lower() != "right" and self.state_tilde.bstate.lower() != "left":
             raise ValueError("Variational state has to have bstate set to 'Right'," \
             "while variational state tilde to 'Left'.")
        
        if PDF.lower() == "left":
            self._sample_fun = self.state.sample
        elif PDF.lower() == "right":
            self._sample_fun = self.state_tilde.sample
        else:
            raise ValueError("PDF has to be either 'Left' or 'Right'.")
        self._PDF = PDF 

        self._ham = self._ham.to_jax_operator()
        self._loss_name_l = "Variance Left"
        self._loss_name_r = "Variance Right"
        self._loss_stats_l = self._loss_stats_r = None
        self.variance = Variance(self._ham, hamiltonian_adj)
        self._eps = None
        self.optimizer_tilde = copy.deepcopy(self._optimizer)
        self.preconditioner_tilde = copy.deepcopy(self._preconditioner)

    @property
    def optimizer_tilde(self):
        return self._optimizer_tilde
    
    @optimizer_tilde.setter
    def optimizer_tilde(self, optimizer):
        self._optimizer_tilde = optimizer
        if optimizer is not None:
            self._optimizer_state_tilde = optimizer.init(self.state.parameters)
            if config.netket_experimental_sharding:
                self._optimizer_state_tilde = jax.lax.with_sharding_constraint(
                    self._optimizer_state_tilde,
                    jax.sharding.PositionalSharding(jax.devices()).replicate(),
                )

    @property
    def preconditioner_tilde(self):
        return self._preconditioner_tilde
    
    @preconditioner_tilde.setter
    def preconditioner_tilde(self, val: PreconditionerT | None):
        if val is None:
            val = identity_preconditioner

        self._preconditioner_tilde = val

    @property
    def PDF(self):
        return self._PDF

    @property
    def state_tilde(self):  
        """
        Returns the left variational state.

        Returns
        -------
        NHMCState
            The left variational state.
        """
        return self._variational_state_tilde
    
    @property
    def eps(self):
        """
        Returns the current estimate of the reference energy ε.

        Returns
        -------
        complex
            The current value of ε used to update the variance loss.
        """
        return self._eps
        
    def _get_eps(self):
        return self.state.nh_expect(self._ham, self.state_tilde, self._PDF, True)
    
    def _trans_step(self):
        alpha = (self._n_fixed + self._n_trans - self._step_count) / self._n_trans 
        self._eps = alpha * self._eps_init + (1 - alpha) * self._get_eps().mean
        self.variance.update(self._eps)
    
    def _self_consistent_step(self):
        self._eps = self._get_eps().mean
        self.variance.update(self._eps)

    @timing.timed
    def _forward_and_backward(self):

        self.state.reset()
        samples = self._sample_fun()
        self.state._samples = samples
        self.state_tilde._samples = samples
        
        if self.step_count < self._n_fixed:
            self._eps = self._eps_init
            self._phase = "Fixed"
        elif self.step_count < self._n_trans + self._n_fixed:
            self._trans_step()
            self._phase = "Transition"
        else:
            self._self_consistent_step()
            self._phase = "Self-consistent"
        
        self._loss_stats_l, self._loss_grad_l = self.state_tilde.expect_and_grad(self.variance.left)
        self._loss_stats_r, self._loss_grad_r = self.state.expect_and_grad(self.variance.right)

        # if it's the identity it does
        # self._dp = self._loss_grad
        self._dp_l = self.preconditioner_tilde(self.state_tilde, self._loss_grad_l, self.step_count)
        self._dp_r = self.preconditioner(self.state, self._loss_grad_r, self.step_count)

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp_l = tree_cast(self._dp_l, self.state_tilde.parameters)
        self._dp_r = tree_cast(self._dp_r, self.state.parameters)
    
    def iter(self, n_steps: int, step: int = 1):
        """
        Returns a generator which advances the VMC optimization, yielding
        after every `step_size` steps.

        Args:
            n_steps: The total number of steps to perform (this is
                equivalent to the length of the iterator)
            step: The number of internal steps the simulation
                is advanced between yielding from the iterator

        Yields:
            int: The current step.
        """
        for _ in range(0, n_steps, step):
            for i in range(0, step):
                self._forward_and_backward()
                if i == 0:
                    yield self.step_count

                self._step_count += 1
                self.update_parameters(self._dp_l, self._dp_r)

    def update_parameters(self, dp_l, dp_r):
        """
        Updates the parameters of the machine using the optimizer in this driver

        Args:
            dp_l: the pytree containing the updates to the left parameters.
            dp_r: the pytree containing the updates to the right parameters.
        """
        self._optimizer_state_tilde, self.state_tilde.parameters = apply_gradient(
            self._optimizer_tilde.update, self._optimizer_state_tilde, dp_l, self.state_tilde.parameters
        )
        self._optimizer_state, self.state.parameters = apply_gradient(
            self._optimizer.update, self._optimizer_state, dp_r, self.state.parameters
        )
    
    def run(
        self,
        n_iter: int,
        eps_init: complex,
        fixed_steps_ratio: float,
        transition_steps_ratio: float,
        out: AbstractLog | Iterable[AbstractLog] | str | None = (),
        obs: dict[str, AbstractObservable] | None = None,
        step_size: int = 1,
        show_progress: bool = True,
        save_params_every: int = 50,  # for default logger
        write_every: int = 50,  # for default logger
        callback: CallbackT | Iterable[CallbackT] = lambda *x: True,
        timeit: bool = False,
    ):
        """
        Runs this variational driver, updating the weights of the network stored in
        this driver for `n_iter` steps and dumping values of the observables `obs`
        in the output `logger`.

        It is possible to control more specifically what quantities are logged, when to
        stop the optimisation, or to execute arbitrary code at every step by specifying
        one or more callbacks, which are passed as a list of functions to the keyword
        argument `callback`.

        Callbacks are functions that follow this signature:

        .. Code::

            def callback(step, log_data, driver) -> bool:
                ...
                return True/False

        If a callback returns True, the optimisation continues, otherwise it is stopped.
        The `log_data` is a dictionary that can be modified in-place to change what is
        logged at every step. For example, this can be used to log additional quantities
        such as the acceptance rate of a sampler.

        Loggers are specified as an iterable passed to the keyword argument `out`. If only
        a string is specified, this will create by default a :class:`nk.logging.JsonLog`.
        To know about the output format check its documentation. The logger object is
        also returned at the end of this function so that you can inspect the results
        without reading the json output.

        When running among multiple MPI ranks/Jax devices, the logging logic is executed
        on all nodes, but only root-rank loggers should write to files or do expensive I/O
        operations.

        .. note::

            Before NetKet 3.15, loggers where automatically 'ignored' on non-root ranks.
            However, starting with NetKet 3.15 it is the responsability of a logger to
            check if it is executing on a non-root rank, and to 'do nothing' if that is
            the case.

            The change was required to work correctly and efficiently with sharding. It will
            only affect users that were defining custom loggers themselves.

        Args:
            n_iter: the total number of iterations to be performed during this run.
            eps_init: initial estiamte of epsilon.
            fixed_steps_ratio: fraction of total iterations to keep ε fixed to eps_init
            transition_steps_ratio: Fraction of total iterations used for interpolating ε between eps_init 
                and the current expectation value.
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
                If this argument is a string, it will be used as output prefix for the standard JSON logger.
            obs: An iterable containing all observables that should be computed
            step_size: Every how many steps should observables be logged to disk (default=1)
            callback: Callable or list of callable callback functions to stop training given a condition
            show_progress: If true displays a progress bar (default=True)
            save_params_every: Every how many steps the parameters of the network should be
                serialized to disk (ignored if logger is provided)
            write_every: Every how many steps the json data should be flushed to disk (ignored if
                logger is provided)
            timeit: If True, provide timing information.
        """

        if not isinstance(n_iter, numbers.Number):
            raise ValueError(
                "n_iter, the first positional argument to `run`, must be a number!"
            )

        self._n_iter = n_iter
        self._n_fixed = int(n_iter * fixed_steps_ratio)
        self._n_trans = int(n_iter * transition_steps_ratio) 
        self._eps_init = self._eps = eps_init
        self.variance.update(self._eps)

        if obs is None:
            obs = {}

        # if out is a path, create an overwriting Json Log for output
        if isinstance(out, str):
            out = JsonLog(out, "w", save_params_every, write_every)
        elif out is None:
            out = ()

        loggers = _to_iterable(out)
        callbacks = _to_iterable(callback)
        callback_stop = False

        with timing.timed_scope(force=timeit) as timer:
            with tqdm(
                total=n_iter,
                disable=not show_progress or not self._is_root,
                dynamic_ncols=True,
            ) as pbar:
                old_step = self.step_count
                first_step = True

                for step in self.iter(n_iter, step_size):
                    log_data = self.estimate(obs)
                    self._log_additional_data(log_data, step)


                    # if the cost-function is defined then report it in the progress bar
                    if self._loss_stats_l is not None and self._loss_stats_r is not None:
                        pbar.set_postfix_str(
                            f"{self._phase} phase | "
                            f"{self._loss_name_l} = {_format_stats(self._loss_stats_l)}, " 
                            f"{self._loss_name_r} = {_format_stats(self._loss_stats_r)}"
                        )
                        log_data[self._loss_name_l] = self._loss_stats_l
                        log_data[self._loss_name_r] = self._loss_stats_r
                        log_data["Eps"] = self._eps

                    # Execute callbacks before loggers because they can append to log_data
                    for callback in callbacks:
                        if not callback(step, log_data, self):
                            callback_stop = True

                    with timing.timed_scope(name="loggers"):
                        for logger in loggers:
                            logger(self.step_count, log_data, self.state)

                    if len(callbacks) > 0:
                        if mpi.mpi_any(callback_stop):
                            break

                    # Reset the timing of tqdm after the first step, to ignore compilation time
                    if first_step:
                        first_step = False
                        pbar.unpause()

                    # Update the progress bar
                    pbar.update(self.step_count - old_step)
                    old_step = self.step_count

                # Final update so that it shows up filled.
                pbar.update(self.step_count - old_step)

        # flush at the end of the evolution so that final values are saved to
        # file
        for logger in loggers:
            logger.flush(self.state)

        if timeit:
            self._timer = timer
            if self._is_root:
                print(timer)

        return loggers