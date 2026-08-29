"""An attribute dereference is fatal on None just as a subscript is, so a single
`info.weight` read at the head of the branch establishes the precondition."""
import torch;

@torch.compile()
def f(x: any, info: any = None) -> any {
    if torch.any(x > 0) {
        fused = info.weight * 2;
    } else {
        fused = None;
    }
    return fused;
}
