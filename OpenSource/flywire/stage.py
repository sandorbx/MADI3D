# SPDX-License-Identifier: AGPL-3.0-only
"""Stage the standalone FlyWire source subset into a new directory for qualification."""

import argparse
from pathlib import Path
from shutil import copy2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    destination = parser.parse_args().destination.resolve()
    source = Path(__file__).resolve().parents[2]
    code = (source / "OpenSource/flywire/source-files.txt").read_text(encoding="utf-8").splitlines()
    support = [
        "constraints-build.txt",
        "OpenSource/LICENSE",
        "OpenSource/README.md",
        "OpenSource/flywire/README.md",
        "OpenSource/flywire/macos-build-constraints.txt",
        "OpenSource/flywire/requirements.txt",
        "OpenSource/flywire/source-files.txt",
        "OpenSource/flywire/stage.py",
        "OpenSource/flywire/smoke.py",
        "LICENSES/THIRD-PARTY.json",
        "LICENSES/THIRD-PARTY.md",
        "LICENSES/VTK-native-NOTICES.txt",
        "LICENSES/pysimdjson-native-NOTICES.txt",
    ]
    destination.mkdir(parents=True)
    for relative in [*code, *support]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(source / relative, target)


if __name__ == "__main__":
    main()
