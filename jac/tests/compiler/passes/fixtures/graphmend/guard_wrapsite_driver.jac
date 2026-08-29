"""A driver program whose wrapped model comes out of a registry lookup.

`torch.compile(BUILDERS["net"]())` is the shape real driver programs take, and
its argument resolves to no definition, so no wrapped entry key can be named
and the model code it pulls in roots its own claim chain. The boundary attach
still happens: `exit_assignment` rewrites every `x = torch.compile(...)`
assignment unconditionally, so the program does hold a place to decode a trap
marker raised inside the compiled region."""
import torch;

glob BUILDERS = {"net": torch.nn.Identity};

glob compiled_guard_net = torch.compile(BUILDERS["net"]());
