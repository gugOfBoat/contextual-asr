#!/usr/bin/env bash
# Tải list IS21 (Le et al. 2021) từ facebookresearch/fbai-speech vào bench/data/is21/
set -e
cd "$(dirname "$0")/is21"
BASE=https://raw.githubusercontent.com/facebookresearch/fbai-speech/main/is21_deep_bias
for s in test-clean test-other; do
  for n in 100 500 1000 2000; do
    f=$s.biasing_$n.tsv
    [ -f "$f" ] || curl -fL -o "$f" "$BASE/ref/$f"
  done
done
for f in all_rare_words.txt common_words_5k.txt; do
  [ -f "$f" ] || curl -fL -o "$f" "$BASE/words/$f"
done
ls -la
