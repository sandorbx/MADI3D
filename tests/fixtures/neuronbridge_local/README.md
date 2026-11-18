# Local positive Color-Depth Search fixtures

`mipsearch.mjs` is the unmodified `search/src/main/nodejs/mipsearch.js` from
[JaneliaSciComp/neuronbridge-services at 3847c602fcc5e3ae2bae47b7308ee11263ae6cef](https://github.com/JaneliaSciComp/neuronbridge-services/blob/3847c602fcc5e3ae2bae47b7308ee11263ae6cef/search/src/main/nodejs/mipsearch.js).
Its SHA-256 is `22f61ae357a4e6c13f5a7c0e1a7a3e3fb576c5c28e13d417f31752ec34dbbc38`.
The upstream BSD 3-Clause license is retained in
[`LICENSES/neuronbridge-services.txt`](../../../LICENSES/neuronbridge-services.txt).
The corresponding `batch_search.js` SHA-256 is
`6455a0183e58fa390c3df02facb4762d6453ca2a472787654f7643c8527a040a`.
It supplies the strict minimum-overlap acceptance rule.

`oracle.json` contains 1,849 RGB-pair gaps and 20 complete query/candidate
cases produced by executing that JavaScript. `generate-oracle.mjs` verifies
the source hash and appends an export in memory to expose its private pixel
function. It does not rewrite the pinned source. Tests compare pixel gaps
within floating-point rounding tolerance and compare integer match counts,
foreground counts, winning mirror state, shift order and denominator behavior
exactly. Empty foreground is rejected explicitly by MADI3D.

The remaining files have deliberately narrow roles. `generated-query.json`
is the sparse RGB query produced by MADI3D for the scorer oracle;
`config.json` and `hemibrain-metadata.json` are bounded public-response
prototypes used to construct explicitly synthetic test libraries;
`public-probe.json` is dated live-service evidence; and `oracle.json` contains
the independently generated routine-test expectations. The public corpus
includes the synthetic query and JavaScript oracle. The bounded service
responses and live probe are retained in the development repository.

To regenerate with a development Node installation available on PATH:

```text
python tests/generate_neuronbridge_local_oracle.py
```

This first generates `generated-query.json` through MADI3D's existing CDM
generator, then executes the pinned JavaScript. Node is used only to regenerate
these committed expectations. Application execution, installation, packaged
smokes and routine tests require no Node, R, Fiji or Java runtime.

`config.json` and `hemibrain-metadata.json` are bounded public responses from
NeuronBridge data release `v3_10_0`. `public-probe.json` records the independent
live probe on 2026-09-12: replica 0 membership manifests, one searchable TIFF
and corresponding resolved identity per enabled collection. It retains source
URLs, sizes, observed validators, downloaded SHA-256 and decoded RGB hashes.
These are locally observed hashes; an S3 ETag is not asserted to be a universal
content checksum, and retained fixture hashes do not guarantee that the live
public service remains unchanged or available. Public NeuronBridge data is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see
[NeuronBridge](https://neuronbridge.janelia.org/) and
[Clements et al. (2024)](https://doi.org/10.1186/s12859-024-05732-7).

Routine installed-library tests derive explicitly synthetic small packages in
the published layout (package manifest, `inventory.sqlite`, sparse shard) from
these prototypes. Counts are changed together with their fixture catalog
declarations; these fixtures are never published as complete public libraries.
The dated `public-probe.json` retains sampled live evidence only; it never
represents a downloaded entire collection.
