"""Input Embedding Package.

This package contains the implementation of the Input Embedding layer,
which is a core component of the Transformer model architecture.
The Input Embedding layer combines an embedding layer and positional encoding
to efficiently handle sequence data in a neural network.

The embedding layer is responsible for converting discrete tokens
(e.g., words or musical notes) into continuous vectors,
allowing the model to learn meaningful representations of the input data.

Positional encoding is necessary because the Transformer model lacks
recurrent or convolutional connections, which means it cannot capture
the position information of the input sequence by default.
By adding positional encoding, the model can effectively learn
the dependencies and relationships between elements in the sequence,
enhancing its expressiveness and generalization capabilities.
"""
