"""Soundness: the branch can return before it reaches the dereference, so a run
that leaves that way finishes with `info` still None. GraphMend must DECLINE."""
import torch;

@torch.compile()
def f(x: any, flag: any = None, info: any = None) -> any {
    if torch.any(x > 0) {
        if flag {
            return torch.zeros(3);
        }
        fused = info["values"];
    } else {
        fused = None;
    }
    return fused;
}
