"""Smallest correct GraphMend measurement, as a template for your own script.

RUN IT LIKE THIS, from THIS directory:

    cd artifact
    PYTHONPATH=../jaseci/jac python -m jaclang run minimal_example.py

Two things about that command line are load-bearing, and getting either wrong
gives you a clean run that measures nothing at all:

  1. `python -m jaclang run`, NOT `python minimal_example.py`.
     GraphMend injects the deferred-side-effect hooks at the
     `torch.compile(...)` assignment site below. That is a source
     transformation, so it only happens in a module Jac compiled. Under plain
     CPython no forward pre-hook is registered, the side-effect depth stays at
     zero, every deferred logger call runs inline, and every logger break
     survives. The model code is genuinely transformed, so nothing looks wrong.

  2. `cd artifact` first, because configuration resolves to the NEAREST
     ANCESTOR jac.toml, and artifact/jac.toml is the one that sets
     `graphmend_claim_imports = true`. Without that opt-in GraphMend does not
     claim imported third-party code, transformers is untouched, and both arms
     measure the same thing. Copy artifact/jac.toml next to your own script if
     you work somewhere else.

This file is a template for your own measurement script. The authoritative
path is paper_eval/, which does the same thing for both arms and compares
them; this one runs a single arm so that the two points above are visible in
isolation.

To see both arms by hand: run it once with artifact/jac.toml as shipped, then
again with `graphmend_claim_imports = false`, and the break count moves from 0
to 3 -- the same t5-small row the harness reports as 3 -> 0. If it does not
move, one of the two points above is not satisfied.
"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from transformers import T5Config, T5ForConditionalGeneration  # noqa: E402

# A small random-weight config. Graph breaks are structural (they are code
# paths), so a small config carries the same breaks as the pretrained model.
# This is emphatically NOT true of latency, which needs the real weights.
torch.manual_seed(0)
config = T5Config(vocab_size=128, d_model=64, d_ff=128, num_layers=2,
                  num_heads=2, d_kv=32)
model = T5ForConditionalGeneration(config).eval()
inputs = {
    "input_ids": torch.randint(0, 128, (1, 8)),
    "decoder_input_ids": torch.randint(0, 128, (1, 8)),
}

# A counting backend: every FX graph Dynamo produces is one entry. Dynamo emits
# one graph for an unbroken region and one more for each break, so
# breaks = graphs - 1.
graphs = []


def counting_backend(gm, example_inputs):
    """Record each FX graph Dynamo hands back, and run it unchanged."""
    graphs.append(gm)
    return gm.forward


torch._dynamo.reset()

# THE SITE THAT MATTERS. GraphMend rewrites this statement to register the
# forward pre-hook and the always-call forward hook that drive [Defer]. Under
# plain `python` this stays a bare torch.compile call.
compiled = torch.compile(model, backend=counting_backend, dynamic=False)

with torch.no_grad():
    out = compiled(**inputs)

print(f"graphs={len(graphs)} breaks={max(0, len(graphs) - 1)}")
print(f"logits checksum={float(out.logits.float().sum()):.6f}")
print()
print("Expect breaks=0 with artifact/jac.toml as shipped.")
print("Expect breaks=3 with graphmend_claim_imports = false, or under plain")
print("python, which is the same t5-small row the harness reports as 3 -> 0.")
