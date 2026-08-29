"""Soundness: `scale` is first defined inside the body and read after the
branch, so it has no value on the false path. Predicating it would read an
unbound name and hoisting it would publish the taken branch's value to runs
that never took it. GraphMend must DECLINE."""
import torch;

@torch.compile()
def f(x: any) -> any {
    out = x * 0;
    if (torch.isnan(x).any()) {
        scale = x * 2;
        out = out + scale;
    }
    return out * scale;
}
