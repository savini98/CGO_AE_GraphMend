"""The accept shape with tensor work on both sides of the guard, so the break
the rule removes is countable.

[Where] declines the merge here because the false operand is a call, which
torch.where would have to evaluate on every run. [Precond] applies instead: the
true branch subscripts `info` before anything else, so `info is not None` is a
necessary condition of reaching the end of it. Driven with `info` left at its
default, the guard folds in Python and the two halves fuse into one graph."""
import torch;

@torch.compile()
def f(x: any, info: any = None) -> any {
    y = x * 2;
    if torch.any(y > 0) {
        out = info["values"] * y;
    } else {
        out = torch.zeros(4);
    }
    return out + 1;
}
