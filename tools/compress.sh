#!/bin/bash
# tools/compress.sh IN OUT  -> two-pass x264 sized for ~28.5 MB (fits a chat upload)
IN=$(realpath "$1"); OUT=$(realpath -m "$2")
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$IN")
vb=$(python3 -c "print(int(28.5*8*1024/$dur - 160))")
cd "$(dirname "$OUT")"; L=p_$(basename "$OUT" .mp4)
ffmpeg -v error -y -i "$IN" -c:v libx264 -preset slow -b:v ${vb}k -pass 1 -passlogfile $L -an -f mp4 /dev/null && \
ffmpeg -v error -y -i "$IN" -c:v libx264 -preset slow -b:v ${vb}k -pass 2 -passlogfile $L -c:a aac -b:a 160k -movflags +faststart "$OUT"
ls -la "$OUT"
