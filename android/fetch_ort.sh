#!/usr/bin/env bash
# Downloads ONNX Runtime for Android (the .aar from Maven Central) and extracts the two pieces
# build.sh needs: libs/onnxruntime.jar and jni/<abi>/*.so. Run once before the first build.
# These are large binaries, so they are not kept in git.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VER=${ORT_VERSION:-1.31.0}
URL="https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$VER/onnxruntime-android-$VER.aar"

mkdir -p "$HERE/libs" "$HERE/jni"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
echo "downloading ONNX Runtime $VER ..."
curl -fsSL -o "$tmp/ort.aar" "$URL"
unzip -o -q "$tmp/ort.aar" classes.jar 'jni/arm64-v8a/*' 'jni/x86_64/*' -d "$tmp/x"
cp "$tmp/x/classes.jar" "$HERE/libs/onnxruntime.jar"
mkdir -p "$HERE/jni/arm64-v8a" "$HERE/jni/x86_64"
cp "$tmp/x/jni/arm64-v8a/"*.so "$HERE/jni/arm64-v8a/"
cp "$tmp/x/jni/x86_64/"*.so   "$HERE/jni/x86_64/"
echo "ok: libs/onnxruntime.jar + jni/{arm64-v8a,x86_64}/*.so  (arm64 = phones, x86_64 = Waydroid)"
