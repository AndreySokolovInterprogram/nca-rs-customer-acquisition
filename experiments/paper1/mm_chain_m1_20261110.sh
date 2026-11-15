#!/bin/bash
# Цепочка прогона статьи (протокол v2, пересчёт M1 от 10.11.2026). Строго по очереди; стоп при первой ошибке.
# Перед запуском: эмбеддинги всех элементов выборки в $PAPER1_MM_WORK/emb_parts + manifest.jsonl
#   (mm_s4_pipeline local … и/или mm_s12_m1_embed media), исходные таблицы — в $PAPER1_MM_WORK/sample_src,
#   sample_stats.json — в $PAPER1_META_DIR (см. README).
# Переменные: PAPER1_MM_WORK (рабочие данные), PAPER1_OUT (результаты), PY (интерпретатор), PAPER1_N_JOBS.
set -u
P=$(cd "$(dirname "$0")/../.." && pwd); cd "$P"
export PYTHONPATH=. HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} PAPER1_N_JOBS=${PAPER1_N_JOBS:-2} PAPER1_ENCODERS=qwen PAPER1_MAX_CLUSTER_SHARE=0.5
PY=${PY:-python}
MMW=${PAPER1_MM_WORK:-$P/work}; export PAPER1_MM_WORK=$MMW
export PAPER1_META_DIR=${PAPER1_META_DIR:-$MMW/meta}
R=${PAPER1_OUT:-$P/results/run}; L=$R/logs; mkdir -p "$R" "$L"
CL=$L/chain.log
st() { echo "$(date '+%F %T') $*" >> "$CL"; }
# возобновление: RESUME_FROM=<имя шага> — предыдущие шаги пропускаются (их результаты уже на диске)
SKIP=${RESUME_FROM:+1}
run() { local name=$1; shift; if [ -n "$SKIP" ]; then [ "$name" = "$RESUME_FROM" ] && SKIP= || { st "skip $name"; return 0; }; fi; st "start $name"; "$@" > "$L/$name.log" 2>&1; local rc=$?; st "end $name rc=$rc"; [ $rc -eq 0 ] || { st "STOP: $name rc=$rc"; exit 3; }; }

run fallback $PY -m experiments.paper1.mm_s12_m1_embed fallback
run finalize $PY -m experiments.paper1.mm_s5_finalize
cp "$PAPER1_META_DIR"/media_stats.json "$PAPER1_META_DIR"/sample_stats.json "$R"/

# основной прогон (ERR ≥ медианы), зерно 42
export PAPER1_WORK_DIR=$MMW/main PAPER1_RESULTS_DIR=$R/main
mkdir -p "$R/main"
for s in 3a 3b 3c 3d 3e; do run s4_$s $PY -m experiments.paper1.s4_protocol --stage $s; done
run s7_main $PY -m experiments.paper1.s7_tables
# чувствительность: сопоставление только по полным профилям корпуса
run sens $PY -m experiments.paper1.mm_s6_checks sens
run check_9b $PY -m experiments.paper1.mm_s6_checks 9b
run check_9a $PY -m experiments.paper1.mm_s6_checks 9a

# 9г: Q3 — тот же энкодер и веса, заново 3в–3д и M1
export PAPER1_WORK_DIR=$MMW/q3 PAPER1_RESULTS_DIR=$R/q3
mkdir -p "$R/q3"; cp "$R/main/stage_3a.json" "$R/main/stage_3b.json" "$R/q3/"
for s in 3c 3d 3e; do run q3_$s $PY -m experiments.paper1.s4_protocol --stage $s; done
run s7_q3 $PY -m experiments.paper1.s7_tables

# 9в: зёрна 43–46 (42 = основной): кластеризация и M1
for seed in 43 44 45 46; do
  W=$MMW/seed$seed; mkdir -p "$W"; ln -sfn "$MMW/main/sample" "$W/sample"; ln -sfn "$MMW/main/emb" "$W/emb"
  export PAPER1_WORK_DIR=$W PAPER1_RESULTS_DIR=$R/seed$seed PAPER1_SEED=$seed
  mkdir -p "$R/seed$seed"; cp "$R/main/stage_3a.json" "$R/main/stage_3b.json" "$R/main/stage_3d.json" "$R/seed$seed/"
  run seed${seed}_3c $PY -m experiments.paper1.s4_protocol --stage 3c
  run seed${seed}_3e $PY -m experiments.paper1.s4_protocol --stage 3e
done
unset PAPER1_SEED

export PAPER1_MM_RESULTS=$R
run checks_tables $PY -m experiments.paper1.mm_s7_checks_tables
export PAPER1_WORK_DIR=$MMW/main PAPER1_RESULTS_DIR=$R/main
run attrition $PY -m experiments.paper1.mm_s13_attrition
run m1_diag $PY -m experiments.paper1.mm_s15_m1_diag
st "ALL DONE"
