# Standalone FlyWire source qualification

Run these commands from the root of the public MADI3D source checkout. Use
native CPython 3.11.9 and activate a fresh virtual environment
(`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on Unix).

```sh
python -m venv .venv
# Activate .venv as described above.
python -m pip install pip==26.2.1
# Windows and Ubuntu
python -m pip install --only-binary=:all: -r OpenSource/flywire/requirements.txt
# macOS Apple Silicon
python -m pip install --only-binary=:all: --no-binary=gevent --build-constraint OpenSource/flywire/macos-build-constraints.txt -r OpenSource/flywire/requirements.txt
# macOS Intel, with OPENSSL_DIR set as described below
python -m pip install --only-binary=:all: --no-binary=gevent,cryptography --build-constraint OpenSource/flywire/macos-build-constraints.txt -r OpenSource/flywire/requirements.txt
python -m pip check
python -I -B OpenSource/flywire/smoke.py
```

The supported targets for qualification are Windows x64, Ubuntu 22.04 x64,
macOS Intel and macOS Apple Silicon. Use a native interpreter. Install the
Xcode Command Line Tools on macOS (`xcode-select --install`). PyPI supplies
gevent 26.9.0 as a source distribution on macOS, so pip builds it with the
pinned build dependencies. Intel macOS also builds cryptography 50.0.2.
Install Rust 1.83 or newer and OpenSSL 3 development files on that target.
With Homebrew OpenSSL, run `export OPENSSL_DIR="$(brew --prefix openssl@3)"`
before installing. The remaining runtime dependencies use matching wheels.
The requirements select the FlyWire runtime from the published MADI3D build
constraints and pin the remaining resolved dependencies.

A private development checkout can stage the same subset in a new directory
using `python OpenSource/flywire/stage.py PATH`. Change into that directory
before creating the environment and running the commands above. The release
workflow does this on each target and records the pip installation report.

## Supported source entry points

| Module under `madi3d_app.integrations.flywire` | Supported use |
| --- | --- |
| `records`, `input_parser` | Fixed-release identity and lossless input parsing |
| `catalog`, `catalog_review`, `catalog_download` | Install local annotation products, search and look up exact source rows, enrich parsed inputs, or download supported products |
| `backend`, `geometry`, `process` | Retrieve native-source geometry in a supervised worker, produce VTP or SWC in micrometres, and verify the resulting artifact |

The [exact standalone source list](source-files.txt) includes shared I/O,
mesh, SWC and VTK rendering helpers used by the detached route. The supported
entry points end at the detached artifact and its retrieval evidence.
MADI3D's application handles scene admission, cache adoption and normalization.

## Code and data inventory

| Material | Terms and provenance | Included in standalone source |
| --- | --- | --- |
| Project code in [source-files.txt](source-files.txt) | MADI3D contributors, AGPL-3.0-only under the [scope notice](../README.md) and [license text](../LICENSE) | Yes |
| [Smoke program and fixtures](smoke.py), [staging program](stage.py) | MADI3D contributors, AGPL-3.0-only. The smoke generates fabricated CSV and Feather rows, a triangle and a two-node skeleton. It uses no upstream records in offline mode. | Generator code only |
| Python distributions and native wheel components | Individually licensed. See the [project dependency inventory](../../LICENSES/THIRD-PARTY.json), [notices](../../LICENSES/THIRD-PARTY.md) and [VTK native notices](../../LICENSES/VTK-native-NOTICES.txt). The pip install report records selected archives, hashes and distribution metadata. Keep the upstream notices shipped with each installation. | Installed separately from PyPI |
| CloudVolume 12.15.2 | Seung Lab and contributors, BSD-3-Clause. See the upstream [LICENSE](https://github.com/seung-lab/cloud-volume/blob/8ba64a2caca8c73985a8bafdbf6f1f278e4b1b5a/LICENSE) and [AUTHORS](https://github.com/seung-lab/cloud-volume/blob/8ba64a2caca8c73985a8bafdbf6f1f278e4b1b5a/AUTHORS). | Installed separately from PyPI |
| Intel macOS cryptography build | cryptography is Apache-2.0 OR BSD-3-Clause. Homebrew's separately installed OpenSSL 3 is Apache-2.0. See the [OpenSSL license](https://www.openssl.org/source/license.html). | Installed or built locally |
| FAFB 783 | FlyWire Consortium and contributing groups. Public release CC BY-NC 4.0. Follow the [FlyWire attribution and citation guidelines](https://home.flywire.ai/guidelines). | Accessed only by explicit retrieval |
| MANC 1.2.1 | HHMI Janelia FlyEM, Cambridge Connectomics Group and Google Research. The qualified geometry product is CC BY 4.0. See the [project](https://www.janelia.org/project-team/flyem/manc-connectome), [dataset paper](https://doi.org/10.7554/eLife.97769.1) and [companion paper](https://doi.org/10.7554/eLife.97766.1). | Accessed only by explicit retrieval |
| MaleCNS 1.0 | HHMI Janelia FlyEM, University of Cambridge Department of Zoology, MRC Laboratory of Molecular Biology and Google Research. The qualified geometry and annotation products are CC BY 4.0. See the [download terms](https://male-cns.janelia.org/download/) and [dataset paper](https://www.cell.com/cell/fulltext/S0092-8674%2826%2900942-6). | Accessed only by explicit retrieval |
| Development extracts and reference assets | The private FlyWire fixture directory contains upstream manifests, Draco fragments, annotations, SWC and normalization evidence. Its source records retain product-specific attribution and changes. Normalization and template assets require their own exact asset terms. | Excluded |

The standalone runtime also installs the following distributions outside the
application's dependency inventory. The pinned archives provide the original
copyright and license notices.

| Distribution | License |
| --- | --- |
| [proto-plus 1.29.0](https://pypi.org/project/proto-plus/1.29.0/) | Apache-2.0 |
| [pox 0.3.7](https://pypi.org/project/pox/0.3.7/) | BSD-3-Clause |
| [pybind11 3.1.0](https://pypi.org/project/pybind11/3.1.0/) | BSD-3-Clause |
| [pytest 9.2.0](https://pypi.org/project/pytest/9.2.0/) | MIT |
| [pluggy 1.7.0](https://pypi.org/project/pluggy/1.7.0/) | MIT |
| [iniconfig 2.3.1](https://pypi.org/project/iniconfig/2.3.1/) | MIT |

## Optional live retrieval

From the same environment, run one dataset at a time.

```sh
python -I -B OpenSource/flywire/smoke.py --live fafb
python -I -B OpenSource/flywire/smoke.py --live manc
python -I -B OpenSource/flywire/smoke.py --live mcns
```

Each live run fetches a fixed-release example mesh and verifies it through
the detached production backend. FAFB and MaleCNS also exercise SWC retrieval.
MANC's qualified representation is mesh because its skeleton source lacks
measured radii. The smoke removes its scratch artifacts on exit and imposes
a per-worker deadline, sampled RSS budget and
payload/output caps. It requires access to the respective public endpoints.
Apply the upstream terms to any data you retrieve independently.
