"""An else-less predicated update whose predicate is already a single tensor
bool, so nothing is left behind as Python-level control flow.

`out` is bound before the branch, so its false operand is the incoming `out`
and the whole `if` collapses to a condition temporary plus one selection."""
import torch;

@torch.compile()
def f(x: any) -> any {
    out = x * 1;
    if (torch.isnan(x).any()) {
        out = x * 2;
    }
    return out;
}
