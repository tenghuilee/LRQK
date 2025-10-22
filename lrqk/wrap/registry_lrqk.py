from opencompass.registry import Registry

import lrqk_attention

LRQK_REGISTRY = Registry("lrqk", locations=["lrqk_attention"])


LRQK_REGISTRY.register_module(
    name="DynamicLRQKCache",
    module=lrqk_attention.DynamicLRQKCache,
    force=True,
)
