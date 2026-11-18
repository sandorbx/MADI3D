# NeuronBridge offline scientific corpus

From the root of the public source checkout, use CPython 3.11 and install
the test dependencies in a fresh virtual environment:

```sh
python -m pip install -r OpenSource/neuronbridge/requirements.txt
python -m pytest @OpenSource/neuronbridge/pytest-args.txt
```

The corpus runs offline. It uses committed reference outputs and in-memory
HTTP responses. Qt, credentials, a production library, Node, Java, and Fiji
are unnecessary for running the tests. The dependency list is for the corpus.
Mesh and SWC rasterization elsewhere in the backend also uses VTK.

Live lookups of public precomputed results require network access to
NeuronBridge's public metadata endpoint and use no credentials. Fetching
remote libraries or images also requires network access. Hosted custom-query
submission is outside the standalone backend's supported operations.

| Coverage | Reference and limits |
| --- | --- |
| Positive CDS | 1,849 pixel-gap probes and 20 score cases captured from pinned NeuronBridge JavaScript. Tests cover matched pixels, threshold and overlap rules, and winning mirror and shift. The optional upstream negative-mask route is outside the positive scorer's scope. |
| Local search | Identical synthetic candidate canvases are searched as dense TIFF and sparse RGB. Tests compare acceptance, scores, winning mirror and shift, and rank and tie order. |
| Color-Depth MIP generation | Retained Fiji/ImageJ outputs cover palette bytes, projection, thresholds, masks, input intensity conversion, and placement on the reference canvas. The one real input is a small, attributed FlyLight template crop. |
| Native gradient behavior | Executed Java probes cover scoring primitives and historical normalization. Seven synthetic Java/ImageJ producer outputs cover Brain and VNC, including annotation clearing, borders, and empty or full signal. The public run does not cover real-candidate gradient parity or full two-stage shape reranking. |
| Public API | Synthetic responses cover versioned EM body and LM line lookups, explicit image identity, CDS and PPPM metrics, incomplete results, and transport failures. The tests make no claim about live availability or hosted search parity. |

The [fixture allowlist](corpus-files.txt) names the exact exported files.
The reference manifests retain the upstream revisions and hashes. Some
gradient manifest rows describe real-image captures that are outside this
export. The public pytest argument file selects the synthetic producer rows
explicitly, and their missing inputs or outputs fail the test.
Gradient execution records retain software versions and command arguments
with machine-specific installation and checkout paths omitted.

The JavaScript oracle comes from neuronbridge-services at
`3847c602fcc5e3ae2bae47b7308ee11263ae6cef` under HHMI's BSD-3-Clause notice.
The gradient references identify colormipsearch and EM_MIP_search separately.
The aligners, scorer and producer notices and the ImageJ disclaimer ship in
the [third-party inventory](../../LICENSES/THIRD-PARTY.md).

Janelia Research Campus, FlyLight Project Team supplied the alignment template
described by Bogovic et al., *An unbiased template of the Drosophila brain and
ventral nerve cord*. The crop is shared under CC BY 4.0. The provenance record
gives the original URL, voxel indices, byte order change, and hashes. The
reference Color-Depth MIP image is derived from that crop. [Publication](https://doi.org/10.1371/journal.pone.0236495) · [License](https://creativecommons.org/licenses/by/4.0/) · [Crop provenance](../../tests/fixtures/neuronbridge_colordepth/template-provenance.json)
