"""Soundness: `info` has no None default, so nothing in the signature says None
is a value the author contemplated for it. The rule only reads parameters
declared optional that way. GraphMend must DECLINE."""
import torch;

@torch.compile()
def f(x: any, info: any) -> any {
    if torch.any(x > 0) {
        fused = info["values"];
    } else {
        fused = None;
    }
    return fused;
}
