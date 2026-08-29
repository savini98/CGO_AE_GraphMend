"""Soundness: only one arm of the inner branch dereferences `info`. A run that
takes the other arm finishes the guard's true branch with `info` still None, so
`info is not None` is not a necessary condition of reaching the end of it and
prepending it would change which arm that run takes. GraphMend must DECLINE."""
import torch;

@torch.compile()
def f(x: any, info: any = None) -> any {
    if torch.any(x > 0) {
        if isinstance(info, dict) {
            fused = info["values"];
        } else {
            fused = torch.zeros(3);
        }
    } else {
        fused = None;
    }
    return fused;
}
