"""Learning rate schedulerfunctions for training music generation models.

The purpose of this file is to provide a collection of learning rate scheduler
that can be easily imported and used in different models and tasks.
"""
import math

import tensorflow as tf


class WarmupCosineDecayScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """This class is a custom learning rate scheduler: warm-up phase & cosine decay phase.

    The warm-up phase linearly increases the learning rate
    from 0 to the maximum learning rate over a specified number of steps.
    After the warm-up phase, the learning rate follows a cosine decay schedule,
    which gradually decreases the learning rate according to a cosine function.

    This technique helps the model to converge faster during the initial phase of training
    and improves the generalization of the model by reducing the learning rate over time.
    """

    def __init__(
        self, max_lr: float = 2.5e-4, warmup_steps: int = 2000, total_steps: int = 10000
    ):
        """Initialize the WarmupCosineDecayScheduler class.

        Args:
            max_lr (float, optional):
                The maximum learning rate.
                Default is 2.5e-4, which is the value used in GPT-2.
            warmup_steps (int, optional):
                The number of steps for the warm-up phase.
                Default is 2000, which is the value used in GPT-2.
            total_steps (int, optional):
                The total number of steps for the learning rate schedule.
                Default is 10000.
        """
        super(WarmupCosineDecayScheduler, self).__init__()
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        """Compute the learning rate based on the current step.

        Args:
            step (int): The current step in the training process.

        Returns:
            float: The learning rate value for the given step.
        """
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        # cos_decay decreases from 1 to 0 as the step progresses.
        cos_decay = 0.5 * (
            1
            + math.cos(
                math.pi
                * (step - self.warmup_steps)
                / (self.total_steps - self.warmup_steps)
            )
        )
        return self.max_lr * cos_decay
