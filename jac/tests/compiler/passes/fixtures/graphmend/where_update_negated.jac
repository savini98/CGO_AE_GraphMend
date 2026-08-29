"""A negated tensor predicate: the `not` has to become a tensor negation for
the same reason the short circuit does, since `torch.where` takes no Python
bool."""
import torch;

@torch.compile()
def f(x: any) -> any {
    out = x * 1;
    if not torch.isnan(x).any() {
        out = x * 2;
    }
    return out;
}
