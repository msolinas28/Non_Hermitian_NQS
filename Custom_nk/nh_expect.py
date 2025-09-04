from functools import partial
import jax
import jax.numpy as jnp
from netket.stats import statistics, Stats

@partial(jax.jit, static_argnames=["apply_fun_l", "apply_fun_r", "PDF"])
def _nh_local_sum_single(O_jax, sigma, apply_fun_l, variables_l, apply_fun_r, variables_r, PDF):
    etas, O_elements = O_jax.get_conn_padded(sigma)
    log_sigma_l = apply_fun_l(variables_l, sigma)
    log_sigma_r = apply_fun_r(variables_r, sigma)

    if PDF.capitalize() == "Left":        
        log_etas = jax.vmap(lambda eta: apply_fun_r(variables_r, eta))(etas)
        ratio_num = jnp.exp(log_etas - log_sigma_l)
        ratio_den = jnp.exp(log_sigma_r - log_sigma_l)
    else:        
        log_etas = jax.vmap(lambda eta: apply_fun_l(variables_l, eta))(etas)
        ratio_num = jnp.exp(log_etas.conj() - log_sigma_r.conj())
        ratio_den = jnp.exp(log_sigma_l.conj() - log_sigma_r.conj()) 

    return jnp.einsum("i, i", ratio_num, O_elements), ratio_den

@partial(jax.jit, static_argnames=["apply_fun_l", "apply_fun_r", "PDF"])
def _batch_nh_local_sum(O, sigmas, apply_fun_l, variables_l, apply_fun_r, variables_r, PDF):
    return jax.vmap(lambda s: _nh_local_sum_single(O, s, apply_fun_l, variables_l, apply_fun_r, variables_r, PDF))(sigmas)

@partial(jax.jit, static_argnames=["apply_fun_l", "apply_fun_r", "PDF"])
def _chunk_nh_local_sum(O, sigmas, apply_fun_l, variables_l, apply_fun_r, variables_r, PDF):
    return jax.vmap(lambda x: _batch_nh_local_sum(O, x, apply_fun_l, variables_l, apply_fun_r, variables_r, PDF))(sigmas)

def get_local_sum(
        O_jax, 
        sigmas, 
        apply_fun_l, 
        variables_l, 
        apply_fun_r, 
        variables_r, 
        PDF="Left", 
        chunk_size=None
        ):
    if PDF.capitalize() not in ["Left", "Right"]:
        raise ValueError("'PDF' can only be 'Left' or 'Right'")

    if len(sigmas.shape) == 1:
        return _nh_local_sum_single(
            O_jax, 
            sigmas, 
            apply_fun_l, 
            variables_l, 
            apply_fun_r, 
            variables_r, 
            PDF
            )

    else:
        sigmas = sigmas.reshape((-1, sigmas.shape[-1]))
        if chunk_size is not None: 
            if sigmas.shape[0] % chunk_size != 0:
                raise ValueError("The chunk size must divede the number of samples!")
            sigmas = sigmas.reshape((sigmas.shape[0] // chunk_size, chunk_size, -1))
            return _chunk_nh_local_sum(
                O_jax, 
                sigmas, 
                apply_fun_l, 
                variables_l, 
                apply_fun_r, 
                variables_r, 
                PDF
                )
        else: 
            return _batch_nh_local_sum(
                O_jax, 
                sigmas, 
                apply_fun_l, 
                variables_l, 
                apply_fun_r, 
                variables_r, 
                PDF
                )

def expect(
    O, 
    sigmas, 
    apply_fun_l, 
    variables_l, 
    apply_fun_r, 
    variables_r, 
    PDF="Left",
    jax_operator=False,
    chunk_size=None
    ):
    """
    Compute the biorthogonal expectation value of a quantum operator.

    This function evaluates the expectation value of a quantum operator between
    a left and right variational state over a set of sampled configurations. It
    supports both left and right probability distributions (PDFs).

    Parameters
    ----------
    O : AbstractOperator
        The operator whose expectation value is to be computed.
    sigmas : array-like
        Sampled configurations over which the expectation is evaluated.
    apply_fun_l : Callable
        Function that applies the left model to a configuration.
    variables_l : PyTree
        Parameters of the left variational state.
    apply_fun_r : Callable
        Function that applies the right model to a configuration.
    variables_r : PyTree
        Parameters of the right variational state.
    PDF : str, optional
        The probability distribution used for sampling configurations. 
        Can be `"Left"` or `"Right"`. Default is `"Left"`.
    jax_operator : bool, optional
        If True, `O` is assumed to already be a JAX operator. Default is False.
    chunk_size : int or None, optional
        Number of samples to process at once for memory efficiency. Default is None
        (process all samples at once).

    Returns
    -------
    complex
        The computed expectation value of the operator over the provided configurations.
    """
    if not jax_operator:
        O = O.to_jax_operator()
    return _expect_jax(O, sigmas, apply_fun_l, variables_l, apply_fun_r, variables_r, PDF, chunk_size)

@partial(jax.jit, static_argnames=["apply_fun_l", "apply_fun_r", "PDF", "chunk_size"])
def _expect_jax(
    O_jax, 
    sigmas, 
    apply_fun_l, 
    variables_l, 
    apply_fun_r, 
    variables_r, 
    PDF="Left",
    chunk_size=None
    ):

    local_sum, ratio_den = get_local_sum(
        O_jax, 
        sigmas, 
        apply_fun_l, 
        variables_l, 
        apply_fun_r, 
        variables_r, 
        PDF, 
        chunk_size
        )
    
    numerator = statistics(local_sum)
    denominator = statistics(ratio_den)
    mean = numerator.mean/(denominator.mean + 1e-8)
    variance = jnp.abs(mean / numerator.mean)**2 * numerator.variance + jnp.abs(mean / denominator.mean) * denominator.variance  

    return Stats(mean=mean, error_of_mean=jnp.sqrt(variance/len(local_sum)), variance=variance)