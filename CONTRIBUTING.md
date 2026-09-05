# Contributing to MADI3D

MADI3D uses a mixed-source model. The main application is developed in a
private repository, while selected reusable scientific components are
designated for public release under AGPL-3.0-only. The exact open-source scope
is defined in `OpenSource/README.md`.

Contributions are welcome by arrangement from developers, researchers, and
collaborators whose work fits the project's scientific and engineering goals.

## Access

The current development repository is private and source access is
invitation-only. Before private repository access is granted, a contributor
must accept the current `CONTRIBUTOR_AGREEMENT.md` in writing. Repository access
is limited to the work for which it was granted and may be revoked when that
work ends.

The fact that a component is designated for open-source publication does not
make unrelated private source or repository material public.

## Contribution requirements

Contributions must:

- be original work or include clear authorization and licensing for any
  third-party material;
- preserve MADI3D's scientific-state, geometry, provenance, and reproducibility
  invariants;
- keep algorithms independent from GUI state where practical;
- preserve cross-platform behavior on Windows, macOS Intel, macOS Apple
  Silicon, and Ubuntu;
- include focused tests for behavior that can be tested automatically;
- avoid compatibility wrappers, duplicate replacement files, versioned source
  copies, hidden legacy routing, and other avoidable maintenance debt; and
- document externally licensed code, data, assets, or binaries before they are
  introduced.

The project maintainers decide whether and when a contribution is accepted,
modified, postponed, rejected, or designated for open-source publication.

## Confidentiality

Private MADI3D source, non-public design material, private issue or pull-request
content, credentials, and unpublished project material must not be disclosed or
redistributed outside the authorized collaboration.

Material expressly published by the MADI3D project is no longer confidential
merely because a private development copy also exists. Publication of one
component does not change the confidentiality of the rest of the repository.

## Licensing

Contributors retain copyright in their original contributions but grant the
rights described in `CONTRIBUTOR_AGREEMENT.md`.

For contributions included in the scientific scope defined by
`OpenSource/README.md`, those rights allow MADI3D to publish the contribution
under AGPL-3.0-only and also to use the same contribution in official
proprietary MADI3D releases under separate project rights. This dual-licensing
model is why the contributor agreement is required even for contributions
intended for the open-source scientific components.

The remaining proprietary MADI3D source and official binary distribution remain
governed by the root `LICENSE`.
