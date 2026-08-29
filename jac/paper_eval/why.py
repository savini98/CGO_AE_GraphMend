"""Print the REASON for each graph break in a registered model.

Table 2 attributes each model's breaks to a cause (DC / LC / VG / DS / DO / TI).
A row that reads "N -> N, 0% fixed" is only meaningful once you know which of
those causes the N breaks actually are: an unfixed logger break is a bug, an
unfixed dynamic-shape break is the paper's declared out-of-scope category.

Like entry.py, this must run under `jac run` for the "on" mode to mean
anything -- under plain CPython the entry program is not Jac-compiled, so
GraphMend never injects the deferred side-effect hooks and every [Defer]
rewrite stays inert. Use the wrapper, which writes the right jac.toml per mode:

    PYTHONPATH=$PWD python -m paper_eval.run_why <model_key> [on|off]
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
