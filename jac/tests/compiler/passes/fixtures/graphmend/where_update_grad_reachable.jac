"""The FP16 overflow clamp in a module that also reaches autograd, which is
every real transformers modeling file: they all read a training flag.

torch.where routes the untaken operand into the backward pass, where a
non-finite local derivative would surface as a NaN gradient the branch never
produced. That hazard exists only while gradients are recorded, so the rewrite
is emitted under a runtime grad-mode test rather than declined outright: Dynamo
guards on grad mode instead of treating it as data, so the inference path
specialises to the predicated variant with no break while the training path
keeps the original branch verbatim."""
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

def run_epoch(x: any) {
    loss = f(x).sum();
    loss.backward();
}
