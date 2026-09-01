"""Per-break cause reporting for one model, run under `jac run`.

Prints each surviving graph break with the reason TorchDynamo gives and the
source location, which is what distinguishes a row that is correctly unfixed
from one that is not.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

if os.environ.get("PAPER_EVAL_DIR"):
    sys.path.insert(0, os.environ["PAPER_EVAL_DIR"])

import torch

from paper_eval.registry import MODELS

key = os.environ["GM_MODEL"]
mode = os.environ.get("GM_MODE", "off")
spec = MODELS[key]

torch.manual_seed(0)
model, inputs = spec["build"]()
model.eval()

torch._dynamo.reset()
with torch.no_grad():
    explanation = torch._dynamo.explain(model)(**inputs)

print(f"{key} [{mode}]: graphs={explanation.graph_count} "
      f"breaks={explanation.graph_break_count}")
for i, reason in enumerate(explanation.break_reasons, 1):
    txt = " ".join(str(reason.reason).split())
    where = ""
    for frame in reversed(getattr(reason, "user_stack", []) or []):
        fname = getattr(frame, "filename", "") or ""
        if "transformers" in fname or "site-packages" not in fname:
            where = f"{fname.split('/')[-1]}:{frame.lineno} in {frame.name}"
            break
    print(f"  {i}. {txt[:150]}")
    if where:
        print(f"     at {where}")
