"""The audio-fusion guard shape from Qwen-Audio-Chat's `QWenModel.forward`.

The guard is data dependent, and Dynamo cannot fold `torch.any`, so it costs a
break on every run, including the text-only ones the false arm exists to serve.
[Where] must decline the merge: the true arm yields per-segment tensors against
a `None` false operand.

[Precond] carries it instead. Every path through the true branch dereferences
`info` before it writes anything an observer can see -- one arm subscripts it,
the other iterates it -- so any run that reaches the end of that branch had
`info` non-None. Leading the guard with `info is not None` therefore selects the
same arm on every such run, and hands Dynamo a Python-level operand it can
settle without tracing the tensor test behind it."""
import torch;

@torch.compile()
def f(x: any, past: any = None, info: any = None) -> any {
    if (past is None) and torch.any(x > 0) {
        pos = torch.where(x > 0);
        span = torch.stack((pos[0], pos[0]), dim=1);
        if isinstance(info, dict) {
            fused = info["values"];
        } else {
            fused = torch.concat([e["values"] for e in info]);
        }
    } else {
        fused = None;
    }
    return fused;
}
