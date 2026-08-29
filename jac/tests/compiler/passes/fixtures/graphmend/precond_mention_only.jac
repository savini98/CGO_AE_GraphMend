"""Soundness: the branch reads `info` for its truth value but never dereferences
it, and `bool(None)` is False rather than an error. Nothing establishes the
precondition. GraphMend must DECLINE."""
import torch;

@torch.compile()
def f(x: any, info: any = None) -> any {
    if torch.any(x > 0) {
        fused = torch.zeros(3) if info else torch.ones(3);
    } else {
        fused = None;
    }
    return fused;
}
