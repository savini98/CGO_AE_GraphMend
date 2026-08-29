"""Model code with a validation guard, compiled after a driver program.

The class binds a torch Module base, so `forward` seeds its own compiled
region and nothing here carries a decorator the boundary guard could ride.
Compiled on its own the guard declines; compiled into a program whose main
module holds an attached `torch.compile` wrap site, the marker has somewhere
to be decoded and the guard lowers."""
import torch;

class WrapsiteNet(torch.nn.Module) {
    def forward(self: any, x: any, mask: torch.Tensor) -> any {
        if not mask.all() {
            raise ValueError("wrapsite mask must be all true");
        }
        return torch.sin(torch.relu(x));
    }
}
