from functools import partial
import jax
import jax.numpy as jnp
from netket.stats import statistics, Stats

@partial(jax.jit, static_argnames=["apply_fun"])
def _nh_local_sum_single(O_jax, sigma, apply_fun, variables):
    etas, O_elements = O_jax.get_conn_padded(sigma)
    log_sigma = apply_fun(variables, sigma).conj()
    log_etas = jax.vmap(lambda eta: apply_fun(variables, eta))(etas)
    ratio_num = jnp.exp(log_etas - log_sigma)
    ratio_den = jnp.exp(-2j * log_sigma.imag)

    return jnp.einsum("i, i", ratio_num, O_elements), ratio_den

@partial(jax.jit, static_argnames=["apply_fun"])
def _batch_nh_local_sum(O, sigmas, apply_fun, variables):
    return jax.vmap(lambda s: _nh_local_sum_single(O, s, apply_fun, variables))(sigmas)

@partial(jax.jit, static_argnames=["apply_fun"])
def _chunk_nh_local_sum(O, sigmas, apply_fun, variables):
    return jax.vmap(lambda x: _batch_nh_local_sum(O, x, apply_fun, variables))(sigmas)

def get_local_sum(O_jax, sigmas, apply_fun, variables, chunk_size=None):
    if len(sigmas.shape) == 1:
        return _nh_local_sum_single(O_jax, sigmas, apply_fun, variables)
    else:
        sigmas = sigmas.reshape((-1, sigmas.shape[-1]))
        if chunk_size is not None: 
            if sigmas.shape[0] % chunk_size != 0:
                raise ValueError("The chunk size must divede the number of samples!")
            sigmas = sigmas.reshape((sigmas.shape[0] // chunk_size, chunk_size, -1))
            return _chunk_nh_local_sum(O_jax, sigmas, apply_fun, variables)
        else: 
            return _batch_nh_local_sum(O_jax, sigmas, apply_fun, variables)

def expect(O, sigmas, apply_fun, variables, jax_operator=False, chunk_size=None):
    """
    Compute the biorthogonal expectation value of a quantum operator, using complex conjugation as 
    the operator that implements the pseudo-hermiticity.

    This function wraps the internal `_expect_jax` function and ensures that the
    operator is in JAX-compatible form if needed.

    Parameters
    ----------
    O : AbstractOperator
        The operator whose expectation value is to be computed.
    sigmas : array-like
        The sampled configurations over which the expectation value is evaluated.
    apply_fun : Callable
        The function that applies the model to a configuration.
    variables : PyTree
        The parameters of the variational state.
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
    return _expect_jax(O, sigmas, apply_fun, variables, chunk_size)

@partial(jax.jit, static_argnames=["apply_fun", "chunk_size"])
def _expect_jax(O_jax, sigmas, apply_fun, variables, chunk_size=None):
    local_sum, ratio_den = get_local_sum(O_jax, sigmas, apply_fun, variables, chunk_size) 
    numerator = statistics(local_sum)
    denominator = statistics(ratio_den)
    
    mean = numerator.mean/(denominator.mean + 1e-8)
    variance = jnp.abs(mean / numerator.mean)**2 * numerator.variance + jnp.abs(mean / denominator.mean) * denominator.variance  

    return Stats(mean=mean, error_of_mean=jnp.sqrt(variance/len(local_sum)), variance=variance)