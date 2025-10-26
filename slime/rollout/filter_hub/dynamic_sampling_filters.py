import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "filter_truncated_samples"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float).std() > 0.0
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def filter_truncated_samples(args, samples: list[Sample], **kwargs):
    """Filter out groups that contain any truncated samples."""
    # Check if any sample in the group is truncated
    has_truncated = any(sample.status == Sample.Status.TRUNCATED for sample in samples)
    
    if has_truncated:
        return DynamicFilterOutput(keep=False, reason="contains_truncated")
    else:
        return DynamicFilterOutput(keep=True)
