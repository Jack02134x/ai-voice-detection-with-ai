#!/usr/bin/env bash
# Turn the YouTube links in ai_link.txt / human_link.txt into training clips.
#
# For every link:
#   1. download the best audio into data_collection/downloads/<ai|human>/<channel>/
#   2. cut it with prepare_audio.py (at most MAX_CLIPS clips, spread over the
#      whole video) straight into audio/<ai|human>/<channel>/<title>_[<id>]/
#   3. delete the downloaded audio once the clips are written
# Videos already processed (listed in done.txt) are skipped, so you can add
# links and run it again. A video that fails keeps its download for a retry.
#
# Usage (from anywhere):
#   ./youtubethings.sh                  # both lists
#   ./youtubethings.sh ai               # only ai_link.txt   (or: human)
#   ./youtubethings.sh --dry-run        # show channel / title / length, download nothing
#   MAX_CLIPS=500 ./youtubethings.sh    # clips per video (default 1000, 0 = no limit)
#
# Link files: one URL per line; blank lines and lines starting with # are ignored.
# Each video must contain only one kind of voice (all human or all AI).
# Afterwards run:  python build_local_csv.py

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
PYTHON="${PYTHON:-$HOME/ai-things/bin/python}"
MAX_CLIPS="${MAX_CLIPS:-1000}"
DOWNLOADS="${DOWNLOADS:-$HERE/downloads}"
DONE_LIST="${DONE_LIST:-$HERE/done.txt}"      # "<class> <video id>" per finished video
FAILED_LOG="${FAILED_LOG:-$HERE/failed.txt}"
LINKS_DIR="${LINKS_DIR:-$HERE}"            # where ai_link.txt / human_link.txt live

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

dry_run=0
classes=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        ai|human)  classes+=("$arg") ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $arg"; usage 1 ;;
    esac
done
[ ${#classes[@]} -eq 0 ] && classes=(ai human)

command -v yt-dlp >/dev/null || { echo "yt-dlp not found"; exit 1; }
[ -x "$PYTHON" ] || { echo "Python not found: $PYTHON (set PYTHON=...)"; exit 1; }

# 11-character video id from youtu.be/, watch?v=, shorts/, live/, embed/ links
video_id() {
    if [[ $1 =~ (youtu\.be/|[?\&]v=|/shorts/|/live/|/embed/)([A-Za-z0-9_-]{11}) ]]; then
        echo "${BASH_REMATCH[2]}"
    fi
}

ok=0; failed=0; skipped=0; clips_before=$(find "$PROJECT/audio" -name '*.wav' 2>/dev/null | wc -l)

for cls in "${classes[@]}"; do
    list="$LINKS_DIR/${cls}_link.txt"
    if [ ! -f "$list" ]; then
        echo "Skipping $cls: $list not found"
        continue
    fi

    # unique video ids, in file order
    ids=()
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"; line="${line//[[:space:]]/}"
        [ -z "$line" ] && continue
        id="$(video_id "$line")"
        if [ -z "$id" ]; then
            echo "Not a YouTube video link, skipped: $line"
            continue
        fi
        [[ " ${ids[*]-} " == *" $id "* ]] || ids+=("$id")
    done < "$list"

    echo
    echo "############################################################"
    echo "$cls: ${#ids[@]} unique video(s) in $(basename "$list")"
    echo "############################################################"

    n=0
    for id in "${ids[@]}"; do
        n=$((n + 1))
        url="https://www.youtube.com/watch?v=$id"
        tag="[$cls $n/${#ids[@]}] $id"

        if grep -qx "$cls $id" "$DONE_LIST" 2>/dev/null; then
            echo "$tag  already done, skipping"
            skipped=$((skipped + 1))
            continue
        fi

        if [ "$dry_run" = 1 ]; then
            yt-dlp --no-playlist --skip-download --no-warnings \
                --print "$tag  %(channel)s | %(title)s | %(duration_string)s" "$url" \
                || echo "$tag  could not read video info"
            continue
        fi

        echo
        echo "$tag  downloading audio..."
        # the final path goes to a file: on stdout it would share a line with the progress bar
        pathfile="$(mktemp)"
        yt-dlp --no-playlist -f 'bestaudio/best' --restrict-filenames \
            -o "$DOWNLOADS/$cls/%(channel)s/%(title).80s_[%(id)s].%(ext)s" \
            --print-to-file after_move:filepath "$pathfile" --quiet --progress "$url"
        echo
        file="$(tail -n 1 "$pathfile")"
        rm -f "$pathfile"
        if [ -z "$file" ] || [ ! -f "$file" ]; then
            echo "$tag  DOWNLOAD FAILED"
            echo "$(date '+%F %T') $cls $id download failed" >> "$FAILED_LOG"
            failed=$((failed + 1))
            continue
        fi
        channel="$(basename "$(dirname "$file")")"
        echo "$tag  -> $(basename "$file")  (channel: $channel)"

        if "$PYTHON" "$HERE/prepare_audio.py" "$file" --to "$cls/$channel" --max-clips "$MAX_CLIPS"; then
            rm -f "$file"
            rmdir "$(dirname "$file")" 2>/dev/null
            echo "$cls $id" >> "$DONE_LIST"
            echo "$tag  done, download deleted"
            ok=$((ok + 1))
        else
            echo "$tag  CLIPPING FAILED, keeping $file"
            echo "$(date '+%F %T') $cls $id clipping failed: $file" >> "$FAILED_LOG"
            failed=$((failed + 1))
        fi
    done
done

[ "$dry_run" = 1 ] && exit 0

clips_after=$(find "$PROJECT/audio" -name '*.wav' 2>/dev/null | wc -l)
echo
echo "============================================================"
echo "Videos done: $ok   failed: $failed   already done: $skipped"
echo "New clips in audio/: $((clips_after - clips_before))"
[ "$failed" -gt 0 ] && echo "Failures are listed in $FAILED_LOG (run again to retry)"
echo "Now run:  python build_local_csv.py"
echo "============================================================"
