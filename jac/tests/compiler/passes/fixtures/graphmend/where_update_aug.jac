"""Soundness: an augmented assignment reads and writes in one step, and the
pass does not resolve which binding the read reaches. GraphMend must
DECLINE."""
import torch;

@torch.compile()
def f(x: any) -> any {
    out = x * 1;
    if (torch.isnan(x).any()) {
        out += x;
    }
    return out;
}
