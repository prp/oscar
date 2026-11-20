#!/usr/bin/env bash
# The release image: Oscar.app (built by `cargo app`) and an Applications link on a compressed disk image,
# target/release/Oscar-<version>.dmg, verified after writing. Usage: scripts/make_dmg.sh [out dir]
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-target/release}
stage=$(mktemp -d -t oscar-dmg)
trap 'rm -rf "$stage"' EXIT
cargo app "$stage" >/dev/null
rm "$stage/oscar" # the command line link `cargo app` leaves beside the app: it would dangle in the image
version=$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$stage/Oscar.app/Contents/Info.plist")
ln -s /Applications "$stage/Applications"
dmg="$out/Oscar-$version.dmg"
rm -f "$dmg"
hdiutil create -quiet -volname Oscar -srcfolder "$stage" -format UDZO -ov "$dmg"
hdiutil verify -quiet "$dmg"
echo "dmg: $dmg written ($(du -h "$dmg" | cut -f1))"
