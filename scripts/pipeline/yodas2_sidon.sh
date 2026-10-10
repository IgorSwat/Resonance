#!/bin/bash
# YODAS2-sidon batches <first>..<last> of shard <lang><shard>: fetch, segment (raw batch deleted after) and filter each
# batch, then fix speakers over all of them. With "score" also ΔCE-score and select (share defaults to 0.33).
#
#   scripts/pipeline/yodas2_sidon.sh pl 000 0 38            -> <lang><shard>_train-<first>-<last>_filtered.csv
#   scripts/pipeline/yodas2_sidon.sh en 001 0 49 score      -> + _scored.csv, _dce_details.csv, _selected.csv
#   scripts/pipeline/yodas2_sidon.sh es 100 25 70 score 0.2
#
# Resumable: every finished step leaves a marker in the run directory, $RUNS/yodas_<lang><shard>_<first>-<last>.
# Scoring needs an Echo checkout ($ECHO) with the student checkpoint its p2_final.json points to. Without scoring,
# select at random afterwards: scripts/select/yodas2_sidon_dce.py --random --share 1.0 (per-speaker bounds 4-64).
set -uo pipefail
[ $# -ge 4 ] || { echo "usage: $0 <lang> <shard> <first> <last> [score [share]]"; exit 2; }
LANG_CODE=$1; SHARD=$2; FIRST=$((10#$3)); LAST=$((10#$4)); SCORE=${5:-}; SHARE=${6:-0.33}
# Checkouts and interpreter; the defaults are the GPU boxes' layout.
R=${RESONANCE:-/workspace/Resonance}; E=${ECHO:-/workspace/Echo}; PY=${PYTHON:-/venv/main/bin/python}
NAME=$(printf "%s%s_train-%05d-%05d" $LANG_CODE $SHARD $FIRST $LAST)
RUN=${RUNS:-/workspace}/yodas_${LANG_CODE}${SHARD}_${FIRST}-${LAST}
OUT=$R/data/processed/YODAS2-Sidon; SEG=$OUT/segments; B=$OUT/batches/$LANG_CODE$SHARD
RAW=$R/data/YODAS2-sidon/$LANG_CODE/$SHARD
AHEAD=3; CHUNK=5000
F=$OUT/${NAME}_filtered.csv; S=$OUT/${NAME}_scored.csv; D=$OUT/${NAME}_dce_details.csv; SEL=$OUT/${NAME}_selected.csv
mkdir -p $RUN $SEG $B
log() { echo "$(date '+%F %T') $*" | tee -a $RUN/run.log; }
batch() { printf train-%05d $1; }

# Downloads run ahead of the GPU, but never more than $AHEAD raw batches wait on disk.
fetcher() {
  for i in $(seq $FIRST $LAST); do
    b=$(batch $i)
    [ -e $RUN/$b.segmented ] && continue
    while [ $(find $RAW -maxdepth 1 -type d -name 'train-*' 2>/dev/null | wc -l) -ge $AHEAD ] && [ ! -d $RAW/$b ]; do sleep 10; done
    if (cd $R && $PY scripts/fetch/yodas2_sidon.py --lang $LANG_CODE --shard $SHARD --first $i > $RUN/$b.fetch.log 2>&1); then
      touch $RUN/$b.fetched
    else
      log "fetch $b FAILED"; touch $RUN/$b.fetch_failed; return 1
    fi
  done
}

log "start $NAME${SCORE:+ (with scoring)}"
fetcher &
for i in $(seq $FIRST $LAST); do
  b=$(batch $i)
  if [ ! -e $RUN/$b.segmented ]; then
    until [ -e $RUN/$b.fetched ] || [ -e $RUN/$b.fetch_failed ]; do sleep 5; done
    [ -e $RUN/$b.fetch_failed ] && { log "abort: no $b"; exit 1; }
    log "segment $b"
    (cd $R && $PY scripts/segment/yodas2_sidon.py --root data/YODAS2-sidon/$LANG_CODE --batch $SHARD/$b --lang $LANG_CODE \
        --output $B/${b}_segments.csv --audio-dir $SEG > $RUN/$b.segment.log 2>&1) || { log "segment $b FAILED"; exit 1; }
    rm -rf $RAW/$b
    touch $RUN/$b.segmented
  fi
  if [ ! -e $RUN/$b.filtered ]; then
    log "filter $b"
    (cd $R && $PY scripts/filter/yodas2_sidon.py --input $B/${b}_segments.csv --audio-dir $SEG \
        --output $B/${b}_filtered.csv > $RUN/$b.filter.log 2>&1) || { log "filter $b FAILED"; exit 1; }
    touch $RUN/$b.filtered
  fi
done
wait

if [ ! -e $RUN/fixup.done ]; then
  log "fixup"
  head -1 $B/$(batch $FIRST)_filtered.csv > $F
  for i in $(seq $FIRST $LAST); do tail -n +2 $B/$(batch $i)_filtered.csv >> $F; done
  (cd $R && $PY scripts/filter/yodas2_sidon_speaker_fixup.py --input $F --output $F --root $SEG \
      > $RUN/fixup.log 2>&1) || { log "fixup FAILED"; exit 1; }
  touch $RUN/fixup.done
fi

if [ "$SCORE" != "score" ]; then
  log "all done (no scoring): $F"
  touch $RUN/all.done
  exit 0
fi

if [ ! -e $RUN/score.done ]; then
  mkdir -p $RUN/chunks
  ls $RUN/chunks/part_??? > /dev/null 2>&1 || tail -n +2 $F | split -l $CHUNK -d -a 3 - $RUN/chunks/part_
  for p in $RUN/chunks/part_???; do
    [ -e $p.scored.csv ] && continue
    log "score $(basename $p)"
    (head -1 $F; cat $p) > $p.csv
    (cd $E && $PY $R/scripts/score/yodas2_sidon_dce.py --model-config echo/configs/p2_final.json \
        --teacher-config echo/configs/teacher_standard.json --input $p.csv --audio-dir $SEG \
        --output $p.tmp.csv --details $p.details.csv > $p.log 2>&1) || { log "score $(basename $p) FAILED"; exit 1; }
    mv $p.tmp.csv $p.scored.csv
  done
  head -1 $RUN/chunks/part_000.scored.csv > $S; head -1 $RUN/chunks/part_000.details.csv > $D
  for p in $RUN/chunks/part_???; do tail -n +2 $p.scored.csv >> $S; tail -n +2 $p.details.csv >> $D; done
  touch $RUN/score.done
fi

log "select"
(cd $R && $PY scripts/select/yodas2_sidon_dce.py --input $S --output $SEL --share $SHARE > $RUN/select.log 2>&1) || { log "select FAILED"; exit 1; }
log "all done: $SEL"
touch $RUN/all.done
