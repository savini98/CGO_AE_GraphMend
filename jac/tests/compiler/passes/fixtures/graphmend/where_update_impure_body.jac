"""Soundness: torch.where evaluates the true operand on every run, so a body
value the pass cannot prove pure would print on runs the original never took.
GraphMend must DECLINE."""
import torch;

def audit(v: any) -> any {
    print("clamped");
    return v;
}

@torch.compile()
def f(x: any) -> any {
    out = x * 1;
    if (torch.isnan(x).any()) {
        out = audit(x * 2);
    }
    return out;
}
