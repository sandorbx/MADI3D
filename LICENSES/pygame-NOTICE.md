# pygame and SDL

MADI3D uses pygame 2.6.1 for controller input. Copyright (C) 2000-2001 Pete
Shinners and the pygame contributors. pygame is distributed under
LGPL-2.1-or-later; `LGPL-2.1.txt` provides the complete license text. Its upstream
source headers permit GNU Library GPL version 2 or later; version 2.1 is the
applicable later license supplied here. The source revision is
`b92c0da9d1b5eba6c813138790fc1ab6cea0252c`:
https://github.com/pygame/pygame/tree/b92c0da9d1b5eba6c813138790fc1ab6cea0252c.

The application retains pygame's Python files outside PyInstaller's PYZ archive,
and native extensions and SDL libraries remain separate files. The upstream
packaging hook's unused FreeSans font (GNU GPL) and sample pygame icon assets are
excluded. This does not change MADI3D's controller behavior.

SDL2 uses the zlib license; see `SDL2-LICENSE.txt`.

Windows uses the reviewed pygame wheel, whose additional SDL image/mixer/font,
FreeType (FTL), JPEG, PNG, TIFF, WebP, zlib, Ogg, Opus, modplug and PortMidi
libraries use permissive terms retained in the bundled notices.

macOS and Linux use pygame 2.6.1 built from source with SDL 2.28.4 only.
Unused optional font/image-extension/mixer/MIDI modules are omitted. SDL audio
and external video dependencies are disabled; controller, joystick, event and
haptic support remain.
macOS retains SDL Cocoa and Apple's system OpenGL support because SDL 2.28.4's
Cocoa backend requires the OpenGL context declaration at build time.
The source delivery includes the original sources, `scripts/build_pygame.py`,
the generated Setup and SDL build configuration.
No upstream wheel audio/system-library closure is included on these platforms.

## Replaceable installation

Use a complete copy of the application and matching Python ABI/architecture.
On Windows/Linux, replace pygame's `.py` files and native extensions under
`MADI3D/_internal/pygame/`; wheel libraries may also be at `_internal/` or
`_internal/pygame.libs/`. On macOS, Python files are in
`MADI3D.app/Contents/Resources/pygame/`, native libraries in
`Contents/Frameworks/pygame/` (including `.dylibs/`). Keep the bundle's symlinks.

Rebuild compatible extensions and libraries from the matching sources using
pygame's retained `setup.py` and `buildconfig/` instructions. Replace the whole
interdependent set when ABI changes require it. Re-sign a modified macOS bundle
as described in `Qt-REPLACEMENT.md`. No MADI3D signing key or separate permission
is needed to replace these libraries or reverse engineer MADI3D to debug them.

`THIRD-PARTY-SOURCES.json` identifies the corresponding source assets delivered
on the same release. `pygame-native-NOTICES.txt` retains original notices,
including notices for upstream wheel components that are not present in every
platform package. The actual package inventory identifies shipped files.
