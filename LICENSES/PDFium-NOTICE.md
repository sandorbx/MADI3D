# PDF rendering and text extraction

MADI3D uses pypdfium2 5.13.0 and its bundled PDFium library. The Python
bindings are available under Apache-2.0 or BSD-3-Clause; PDFium and its
included third-party components have their own notices.

Each application package retains the complete installed pypdfium2 wheel
metadata, including `pypdfium2-5.13.0.dist-info/licenses/`. That directory
contains the binding license texts and the platform-specific
`data/<platform>/BUILD_LICENSES/` directory for the actual bundled binary.
These upstream files must accompany redistribution. The package build
checks that the PDFium notice is present.

Upstream: https://github.com/pypdfium2-team/pypdfium2

This dependency requires macOS 13 or newer for the macOS packages.
