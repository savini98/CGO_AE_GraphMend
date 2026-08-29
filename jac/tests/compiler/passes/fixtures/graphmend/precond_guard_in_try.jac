"""Soundness: the guard sits inside a try that catches the very error the
dereference raises, so raising is live control flow here rather than a bug.
GraphMend must DECLINE."""
import torch;

@torch.compile()
def f(x: any, info: any = None) -> any {
    fused = None;
    try {
        if torch.any(x > 0) {
            fused = info["values"];
        } else {
            fused = None;
        }
    } except TypeError {
        fused = torch.zeros(3);
    }
    return fused;
}
