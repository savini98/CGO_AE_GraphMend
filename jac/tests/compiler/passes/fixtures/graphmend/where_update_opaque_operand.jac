"""Soundness: destroying the short circuit needs every operand to be a proven
tensor bool. `flag` is a Python bool, so lowering the chain to a tensor
disjunction would trade a working branch for a TypeError. GraphMend must
DECLINE."""
import torch;

@torch.compile()
def f(x: any, flag: bool) -> any {
    out = x * 1;
    if (flag or torch.isnan(x).any()) {
        out = x * 2;
    }
    return out;
}
