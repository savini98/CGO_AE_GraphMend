"""Soundness: the false operand of the selection is the variable's incoming
value, and `mask` is an optional parameter whose default is None. Selecting
against None is a TypeError, and the enclosing guard proves it is None on
exactly the path the branch sits on. GraphMend must DECLINE.

Taken from timm's Gemma4 ViT forward, which resolves an optional patch mask
this way."""
import torch;

@torch.compile()
def f(x: any, mask: any = None) -> any {
    if mask is None {
        sentinel = torch.isnan(x);
        if sentinel.any() {
            mask = sentinel;
        }
    }
    return mask;
}
