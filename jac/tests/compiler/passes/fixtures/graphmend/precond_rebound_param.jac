"""Soundness: `info` is rebound before the guard, so the value the dereference
proves non-None is not the value the prepended test would read. GraphMend must
DECLINE."""
import torch;

@torch.compile()
def f(x: any, info: any = None) -> any {
    if info is None {
        info = {"values": torch.zeros(3)};
    }
    if torch.any(x > 0) {
        fused = info["values"];
    } else {
        fused = None;
    }
    return fused;
}
