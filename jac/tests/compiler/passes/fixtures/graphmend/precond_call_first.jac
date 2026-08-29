"""Soundness: an opaque call runs before the dereference, so the branch is not
effect free up to the point that proves the precondition. GraphMend must
DECLINE."""
import torch;

def prep(v: any) -> any {
    print("prepping");
    return v * 2;
}

@torch.compile()
def f(x: any, info: any = None) -> any {
    if torch.any(x > 0) {
        staged = prep(x);
        fused = info["values"];
    } else {
        fused = None;
    }
    return fused;
}
