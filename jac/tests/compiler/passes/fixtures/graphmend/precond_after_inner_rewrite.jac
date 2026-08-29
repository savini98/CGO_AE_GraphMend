"""Soundness: the inner branch is one [Where] rewrites, and doing so lifts
`info["a"]` out of the arm that guarded it and onto the straight line. Reading
that lifted dereference as proof of a precondition would be circular: in the
program as written, a run with `info` None and no NaN takes the inner else and
finishes the outer branch without raising. The rule therefore refuses any region
that already carries a synthesized temporary. GraphMend must DECLINE."""
import torch;

@torch.compile()
def f(x: any, info: any = None) -> any {
    if torch.any(x > 0) {
        if torch.isnan(x).any() {
            y = info["a"];
        } else {
            y = x * 2;
        }
        out = y + 1;
    } else {
        out = torch.zeros(4);
    }
    return out;
}
