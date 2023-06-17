"""Add&Norm module for the Transformer model.

This custom Keras layer implements the Add & Norm layer.
It consists of adding a residual connection
(element-wise addition of the input and output from the previous layer)
and applying Layer Normalization.
"""
from typing import Optional, Tuple, Union

import tensorflow as tf


class AddAndNorm(tf.keras.layers.Layer):
    """This class is a custom Keras layer that implements the Add & Norm layer.

    The Add & Norm operation consists of adding the residual connection
    (element-wise addition of the input and output from the previous layer)
    and applying Layer Normalization.
    Residual connections help improve gradient flow
    and mitigate the vanishing gradient problem in deep networks,
    while Layer Normalization stabilizes the learning process
    by normalizing the input across the features.

    It takes the input tensor and the output tensor from the previous layer,
    applies Dropout to the previous layer's output,
    adds the input and the Dropout output element-wise (residual connection),
    and applies LayerNormalization.
    """

    def __init__(self, eps: float = 1e-6, dropout_rate: float = 0.1):
        """Initialize the AddAndNorm class.

        Args:
            eps (float, optional):
                A small constant for numerical stability
                of the LayerNormalization. Default is 1e-6.
            dropout_rate (float, optional):
                The dropout rate to be applied
                after the LayerNormalization. Default is 0.1.
        """
        super(AddAndNorm, self).__init__()
        # The default initial values are gamma=1 and beta=0.
        # So the explicit definition of gamma and beta variables is omitted.
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=eps)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(
        self,
        prev_layer_input: tf.Tensor,
        prev_layer_output: tf.Tensor,
        training: Optional[bool] = None,
        return_prev_layer_output: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Compute the Add&Norm layer for the input tensors.

        This function takes the input and the output from the previous layer,
        applies Dropout to the previous layer's output,
        adds the input and the Dropout output element-wise (residual connection),
        and applies LayerNormalization.
        When calling this function as a layer in the Transformer model,
        don't specify the `return_prev_layer_output` argument,
        so that only a single output value is returned and can be utilized.

        Args:
            prev_layer_input (tf.Tensor):
                The input tensor from the previous layer
                with shape (batch_size, seq_len, d_model).
            prev_layer_output (tf.Tensor):
                The output tensor from the previous layer
                with shape (batch_size, seq_len, d_model).
            training (Optional[bool], optional):
                True if the model is in training mode, False if in inference mode.
                If `None`, the mode will be inferred from `self.training`.
                Default is None.
            return_prev_layer_output (bool, optional):
                If True, the function will return the previous layer's output.
                This is mainly used for testing purposes
                to ensure that Dropout is applied correctly. Default is False.

        Returns:
            Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
                The output tensor with shape (batch_size, seq_len, d_model)
                after adding the input tensors, applying LayerNormalization.
                If `return_prev_layer_output` is True,
                a tuple containing the output tensor and the previous layer output.
        """
        prev_layer_output = self.dropout(prev_layer_output, training=training)
        added = prev_layer_input + prev_layer_output
        output = self.layer_norm(added)
        if return_prev_layer_output:
            return output, prev_layer_output
        else:
            return output
