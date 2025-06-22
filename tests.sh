#!/usr/bin/env bash
set -uuo pipefail   # solo -u, -o pipefail; quitamos el -e

EXEC="./GameOfLife"
TIMEOUT_CMD="timeout -s9 10m"

# Captura la salida ignorando el fallo de timeout
run_test() {
  local exp=$1 mode=$2 X=$3 Y=$4 is2d=$5 dim=$6
  # forzamos que no termine el script aunque timeout “falle”
  local out
  out=$(
    $TIMEOUT_CMD "$EXEC" 0 "$X" "$Y" "$mode" "$is2d" "$dim" \
      2>&1 \
    || true
  )
  # extraemos el último Cells per second
  local cps
  cps=$(printf "%s\n" "$out" \
        | grep "Cells per second" \
        | tail -n1 \
        | awk '{print $4}')
  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "$exp" "$mode" "$X" "$Y" "$is2d" "$dim" "${cps:-TIMEOUT}"
}

# CSV header
echo "EXP,mode,X,Y,is2d,dim,CellsPerSec"

# --- EXP1 ejemplo reducido
EXP="EXP1"
for mode in CPU CUDA OPENCL; do
  for s in 100 500 1500 5000 15000; do
    run_test "$EXP" "$mode" "$s" "$s" 0 32
  done
done

##########
# EXP 2: CUDA & OPENCL, 1d (is2d=0), block size fijo=32
#       tamaños: 100, 256, 512, 1024, 2048, 4096, 8192
##########
EXP="EXP2"
modes=(CUDA OPENCL)
# paso en potencias de 2 desde 2^2 hasta 2^13 (100≈2^7)
sizes=(128 512 1024 2048 4096 8192)
for mode in "${modes[@]}"; do
  for s in "${sizes[@]}"; do
    run_test "$EXP" "$mode" "$s" "$s" 0 32
  done
done

##########
# EXP 3: CUDA & OPENCL, block dims MULTIPLOS vs NO-MULTIPLOS de 32
##########
EXP="EXP3"
modes=(CUDA OPENCL)
# bloques múltiplos de 32
dims_mul=(32 64 96 128 160 192)
# bloques NO-múltiplos de 32
dims_nomul=(30 60 90 120 150 190)
sizes=(8192)

for mode in "${modes[@]}"; do
  # primero múltiplos
  for dim in "${dims_mul[@]}"; do
    for s in "${sizes[@]}"; do
      run_test "$EXP" "$mode" "$s" "$s" 0 "$dim"
    done
  done
  # luego no múltiplos
  for dim in "${dims_nomul[@]}"; do
    for s in "${sizes[@]}"; do
      run_test "$EXP" "$mode" "$s" "$s" 0 "$dim"
    done
  done
done

##########
# EXP 4: CUDA vs OPENCL, block dims [1,2,4,8,16], comparando 1d vs 2d
##########
EXP="EXP4"
modes=(CUDA OPENCL)
dims=(1 2 4 8 16)
sizes=(8192)

for mode in "${modes[@]}"; do
  for is2d in 0 1; do
    for dim in "${dims[@]}"; do
      for s in "${sizes[@]}"; do
        run_test "$EXP" "$mode" "$s" "$s" "$is2d" "$dim"
      done
    done
  done
done
