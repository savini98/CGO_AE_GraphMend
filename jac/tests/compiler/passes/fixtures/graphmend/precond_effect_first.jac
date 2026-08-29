"""Soundness: the branch writes through `holder` before it dereferences `info`.
On a run with `info` None the original leaves that write behind and then raises;
skipping the branch would lose it. GraphMend must DECLINE."""
import torch;

@torch.compile()
def f(x: any, holder: any, info: any = None) -> any {
    if torch.any(x > 0) {
        holder.cache = 1;
        fused = info["values"];
    } else {
        fused = None;
    }
    return fused;
}
