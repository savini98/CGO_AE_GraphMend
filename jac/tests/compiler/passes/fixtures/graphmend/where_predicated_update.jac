"""An else-less data-dependent guard whose body only assigns: the FP16
overflow-clamp idiom carried verbatim by Florence2EncoderLayer.forward and by
the BART/M2M/Marian encoder layers in transformers.

Two things make it distinct from the two-branch shapes. The `if` has no `else`,
so the false value of each assigned variable is its value before the branch,
and the predicate is a short-circuit chain whose `or` over two 0-dim bool
tensors is itself a graph break -- hoisting the condition unchanged trades one
break for another instead of removing it.

`clamp_value` is a body-local temporary read only inside the body, so it hoists
unconditionally; only `hidden_states`, bound on entry, needs predication, and
its false operand is the incoming `hidden_states`."""
import torch;

@torch.compile()
def f(hidden_states: any) -> any {
    hidden_states = hidden_states * 2;
    if (hidden_states.dtype == torch.float16)
        and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()) {
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000;
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value);
    }
    return hidden_states;
}
