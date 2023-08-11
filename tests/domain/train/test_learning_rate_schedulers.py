"""Tests for the Learning Rate Scheduler."""
import math

from generative_music.domain.train.learning_rate_schedulers import \
    WarmupCosineDecayScheduler


class TestWarmupCosineDecayScheduler:
    def setup_method(self):
        """Initialize the input tensors for the WarmupCosineDecayScheduler tests.

        This fixture creates predefined custom scheduler.
        """
        self.scheduler = WarmupCosineDecayScheduler()
        self.max_lr = self.scheduler.max_lr
        self.warmup_steps = self.scheduler.warmup_steps
        self.total_steps = self.scheduler.total_steps

    def test_warmup_phase(self):
        """Check the warm-up phase of the WarmupCosineDecayScheduler.

        Tests if the WarmupCosineDecayScheduler computes
        the expected learning rate values during the warm-up phase,
        where the learning rate linearly increases
        from 0 to the maximum learning rate over a specified number of steps.
        """
        for step in range(self.warmup_steps):
            expected_lr = self.max_lr * step / self.warmup_steps
            assert math.isclose(self.scheduler(step), expected_lr, rel_tol=1e-9)

    def test_cosine_decay_phase(self):
        """Check the cosine decay phase of the WarmupCosineDecayScheduler.

        Tests if the WarmupCosineDecayScheduler computes
        the expected learning rate values during the cosine decay phase,
        where the learning rate gradually decreases
        according to a cosine function after the warm-up phase.
        """
        for step in range(self.warmup_steps, self.total_steps):
            cos_value = math.cos(
                math.pi
                * (step - self.warmup_steps)
                / (self.total_steps - self.warmup_steps)
            )
            cos_decay = 0.5 * (1 + cos_value)
            expected_lr = self.max_lr * cos_decay
            assert math.isclose(self.scheduler(step), expected_lr, rel_tol=1e-9)
