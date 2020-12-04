from typing import Any, Tuple

import jax
import mcx
import jax.numpy as np

from mcx.sample import validate_model_args

__all__ = ["sample_predictive"]


def sample_predictive(
    rng_key: jax.random.PRNGKey,
    model: mcx.model,
    model_args: Tuple[Any],
    num_samples: int = 10,
):
    """Provides a unified interface for prior and posterior predictive sampling.

    The rationale behind using the same API for both prior and posterior
    predictive sampling is simple: both are prediction of the outputs of the
    generative models. Before and after the model has been evaluated on data.

    Note
    ----
    The clunkiness of any prior-posterior predictive sampling API comes from
    the 'static model + trace' abstraction and would disappear within an
    'evaluated program' paradigm.

    Parameters
    ----------
    rng_key
        The key used to seed JAX's random number generator.
    model
        The model that is used for predictions.
    trace
        Samples from the model's posterior distribution computed with the data
        used for predictions or with another dataset. If specified `predict` will
        return posterior predictive samples, otherwise prior predictive samples.

    Returns
    -------
    Either the prior predictive or posterior predictive class.

    """

    """Generate samples from the prior predictive distribution.

    The input data is broadcasted using numpy's broadcasting rules, and we
    then draw `num_samples` samples from the prior predictive distribution
    for each data point so the resulting array is of shape
    `(data_broadcasted_shape, num_samples)`.

    Parameters
    ----------
    num_samples
        The number of prior samples to draw from the posterior predictive
        distribution.
    **observations
        The values of the model's input parameters for which we want to compute
        predictions.

    Returns
    -------
    A dictionary that maps each returned variable to an array of shape
    (data_shape, num_samples) that contains samples from the models' prior
    predictive distribution.

    """
    _ = validate_model_args(model, model_args)
    keys = jax.random.split(rng_key, num_samples)

    in_axes: Tuple[int, ...] = (0,)
    sampler_args: Tuple[Any, ...] = (keys,)
    for arg in model_args:
        try:
            sampler_args += (np.atleast_1d(arg),)
        except RuntimeError:
            sampler_args += (arg,)
        in_axes += (None,)

    # TODO(remi): Handle the case where multiple values are returned.

    samples = jax.vmap(model.call_fn, in_axes, out_axes=1)(*sampler_args)

    return samples
