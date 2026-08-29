"""Soundness: the guard itself calls something not proven effect free. On a run
with `info` None the original still evaluates it, and the rewrite would skip it.
GraphMend must DECLINE."""
import torch;

def probe(v: any) -> any {
    print("probed");
    return v;
}

@torch.compile()
def f(x: any, info: any = None) -> any {
    if torch.any(probe(x) > 0) {
        fused = info["values"];
    } else {
        fused = None;
    }
    return fused;
}
