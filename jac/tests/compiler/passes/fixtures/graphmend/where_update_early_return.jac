"""Soundness: a predicated update has no place to put an early return, whose
whole effect is to skip the rest of the function. GraphMend must DECLINE."""
import torch;

@torch.compile()
def f(x: any) -> any {
    out = x * 1;
    if (torch.isnan(x).any()) {
        out = x * 2;
        return out;
    }
    return out;
}
