# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Kirk's approximation for European spread-option price under Black-Scholes model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp


def spread_option_price(forwards_asset1,
                        volatilities_asset1,
                        forwards_asset2,
                        volatilities_asset2,
                        expiries,
                        strikes,
                        rho,
                        discount_factors=None,
                        is_call_options=None,
                        dtype=None,
                        name=None):
    """Computes the Black Scholes price for a batch of European options.

    ## References:
    [1]C.F.Lo, A simple derivation of Kirk’s approximation for spread options
    https://www.sciencedirect.com/science/article/pii/S0893965913001171

    Args:
      forwards_asset1: A real `Tensor` of any shape. The current forward prices to
        expiry for the first underlying asset,
      volatilities_asset1: A real `Tensor` of same shape and dtype as `forwards_asset1`. The
        volatility to expiry for the first underlying asset.
      forwards_asset2: A real `Tensor` of the same shape and dtype as `forwards_asset1`. The
        current forward prices toexpiry for the second underlying asset, for two underlying options.
      volatilities_asset2: A real `Tensor` of same shape and dtype as `forwards_asset1`. The
        volatility to expiry for the second underlying asset.
      expiries: A real `Tensor` of same shape and dtype as `forwards_asset1`. The expiry
        for each option with the second underlying asset. The units should be such that
         `expiries * volatilities_asset1**2` is dimensionless.
      strikes: A real `Tensor` of the same shape and dtype as `forwards_asset1`. The
        strikes of the options to be priced.
      rho: A real 'Tenor' of the same and dtype as 'forwards_asset1'. The correlation between two
        underlying assets.
      discount_factors: A real `Tensor` of same shape and dtype as the `forwards_asset1`.
        The discount factors to expiry (i.e. e^(-rT)). If not specified, no
        discounting is applied (i.e. the undiscounted option price is returned).
        Default value: None, interpreted as discount factors = 1.
      is_call_options: A boolean `Tensor` of a shape compatible with `forwards_asset1`.
        Indicates whether to compute the price of a call (if True) or a put (if
        False). If not supplied, it is assumed that every element is a call.
      dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
        of any supplied non-`Tensor` arguments to `Tensor`.
        Default value: None which maps to the default dtype inferred by TensorFlow
        (float32).
      name: str. The name for the ops created by this function.
        Default value: None which is mapped to the default name `spread_option_price`.

    Returns:
      spread_option_prices: A `Tensor` of the same shape as `forwards`. The kirk's
      approximation for Black Scholes price of the spread_options.


    #### Examples
    ```python
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    expiries = 1.0
    computed_prices = option_price(
        forwards,
        strikes,
        volatilities,
        expiries,
        dtype=tf.float64)
    # Expected print output of computed prices:
    # [ 0.          2.          2.04806848  1.00020297  2.07303131]
    ```
    """
    with tf.compat.v1.name_scope(
            name,
            default_name='spread_option_price',
            values=[
                forwards_asset1, volatilities_asset1,
                forwards_asset2, volatilities_asset2,
                expiries, strikes, discount_factors, is_call_options
            ]):
        forwards_asset1 = tf.convert_to_tensor(forwards_asset1, dtype=dtype,
                                               name='forwards_asset1')
        volatilities_asset1 = tf.convert_to_tensor(
            volatilities_asset1, dtype=dtype, name='volatilities_asset1')
        forwards_asset2 = tf.convert_to_tensor(forwards_asset2, dtype=dtype,
                                               name='forwards_asset2')
        volatilities_asset2 = tf.convert_to_tensor(
            volatilities_asset2, dtype=dtype, name='volatilities_asset2')
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')

        strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
        if discount_factors is None:
            discount_factors = 1
        discount_factors = tf.convert_to_tensor(
            discount_factors, dtype=dtype, name='discount_factors')
        normal = tfp.distributions.Normal(
            loc=tf.zeros([], dtype=forwards_asset1.dtype), scale=1)

        sqrt_volatilities = tf.math.sqrt(volatilities_asset1 ** 2 - 2 * rho *
                                         volatilities_asset1 *
                                         volatilities_asset2 +
                                         volatilities_asset2 ** 2)
        sqrt_var = sqrt_volatilities * tf.math.sqrt(expiries)
        d1 = (tf.math.log(forwards_asset1) - tf.math.log(forwards_asset2 +
                                                         strikes
                                                         *discount_factors)) \
             / sqrt_var + 0.5 * sqrt_var
        d2 = d1 - sqrt_var
        undiscounted_calls = forwards_asset1 * normal.cdf(d1) - \
                             (forwards_asset2 + strikes * discount_factors) *\
                             normal.cdf(d2);


        undiscounted_calls = forwards * normal.cdf(d1) - strikes * normal.cdf(
            d2)
        if is_call_options is None:
            return discount_factors * undiscounted_calls
        undiscounted_forward = forwards - strikes
        undiscounted_puts = undiscounted_calls - undiscounted_forward
        return discount_factors * tf.where(is_call_options, undiscounted_calls,
                                           undiscounted_puts)


