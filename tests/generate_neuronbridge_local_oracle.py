"""Development-only fixture generator; requires Node to execute the pinned oracle.

Run: python tests/generate_neuronbridge_local_oracle.py
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from madi3d_app.integrations.neuronbridge.cdm import generate_brain_cdm
from madi3d_app.integrations.neuronbridge.cdm_records import CDMMapping
from test_neuronbridge_cdm import arguments


def generated_query():
    data = np.zeros((4, 4, 6), np.uint8)
    data[0, 0, 0], data[1, 1, 2], data[2, 2, 5], data[3, 3, 3] = 80, 180, 240, 255
    kwargs = arguments(data)
    matrix = np.eye(4)
    matrix[:3, 3] = (402, 140, 50)
    kwargs["mapping"] = CDMMapping(matrix, "local-reference-fixture")
    return generate_brain_cdm(data, **kwargs)


if __name__ == "__main__":
    folder = Path(__file__).parent / "fixtures" / "neuronbridge_local"
    result = generated_query()
    y, x = np.nonzero(np.any(result.rgb != 0, axis=2))
    value = {"shape": list(result.rgb.shape[:2]), "pixels": [[int(xx), int(yy), *map(int, result.rgb[yy, xx])] for yy, xx in zip(y, x)]}
    (folder / "generated-query.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    subprocess.run(["node", str(folder / "generate-oracle.mjs")], check=True)
