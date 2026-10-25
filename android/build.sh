#!/usr/bin/env bash
# Builds VoiceCheck.apk without Gradle: aapt2 -> javac -> d8 -> package -> zipalign -> apksigner.
# Bundles ONNX Runtime (libs/onnxruntime.jar + jni/<abi>/*.so) and the model (assets/).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SDK=${ANDROID_HOME:-/home/jack/Projects/android_apps/android-sdk}
BT=$SDK/build-tools/37.0.0
JAR=$SDK/platforms/android-36/android.jar
ORT=$HERE/libs/onnxruntime.jar

OUT=$HERE/build
APK=$HERE/VoiceCheck.apk
KS=$HERE/release.keystore

for f in "$BT/aapt2" "$BT/d8" "$BT/zipalign" "$BT/apksigner" "$JAR" "$ORT" "$HERE/assets/aasist.onnx"; do
  [[ -e $f ]] || { echo "missing: $f" >&2; exit 1; }
done

rm -rf "$OUT"
mkdir -p "$OUT/gen" "$OUT/classes" "$OUT/dex"

"$BT/aapt2" compile --dir "$HERE/res" -o "$OUT/res.zip"
# versionCode bumps every build (minutes since 2025-01-01) so an update always installs over the
# old app; versionName shows the build date.
VCODE=$(( ($(date +%s) - 1735689600) / 60 ))
VNAME=$(date +%Y.%m.%d.%H%M)
"$BT/aapt2" link -o "$OUT/base.apk" -I "$JAR" \
  --manifest "$HERE/AndroidManifest.xml" -A "$HERE/assets" --java "$OUT/gen" \
  --min-sdk-version 26 --target-sdk-version 34 --version-code "$VCODE" --version-name "$VNAME" \
  "$OUT/res.zip"

javac --release 11 -nowarn -Xlint:-options -classpath "$JAR:$ORT" -d "$OUT/classes" \
  $(find "$HERE/src" "$OUT/gen" -name '*.java')
# dex the app classes together with the ONNX Runtime Java library
"$BT/d8" --release --lib "$JAR" --min-api 26 --output "$OUT/dex" \
  $(find "$OUT/classes" -name '*.class') "$ORT"

cp "$OUT/base.apk" "$OUT/unsigned.apk"
(cd "$OUT/dex" && zip -q "$OUT/unsigned.apk" classes.dex)
# native libraries: lib/<abi>/*.so  (extractNativeLibs=true, so plain deflate is fine)
(cd "$HERE" && mkdir -p "$OUT/lib" && cp -r jni/* "$OUT/lib/" && cd "$OUT" && zip -q -r unsigned.apk lib)

"$BT/zipalign" -p -f 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"

if [[ ! -f $KS ]]; then
  keytool -genkeypair -keystore "$KS" -storepass voicecheck -keypass voicecheck -alias voicecheck \
    -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Voice Check" >/dev/null
fi
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:voicecheck --key-pass pass:voicecheck \
  --out "$APK" "$OUT/aligned.apk"
"$BT/apksigner" verify "$APK" >/dev/null

echo "built $APK ($(du -h "$APK" | cut -f1))"
