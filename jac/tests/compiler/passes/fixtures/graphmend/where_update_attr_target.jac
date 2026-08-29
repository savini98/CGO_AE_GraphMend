"""Soundness: an attribute write is observable outside the region, so it is
not a predicated update of a local. GraphMend must DECLINE."""
import torch;

@torch.compile()
def f(holder: any, x: any) -> any {
    out = x * 1;
    if (torch.isnan(x).any()) {
        holder.cache = x * 2;
    }
    return out;
}
