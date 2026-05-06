#!/bin/bash
echo "N, Serial(s), SIMD(s), Speedup" > results.csv
for n in 256 512 1024 1536 2048; do
    echo -n "Testing n=$n ... "
    serial_time=$(./gauss_serial $n 2>/dev/null | grep -oP 'time = \K[0-9.]+')
    simd_time=$(./gauss_simd $n 2>/dev/null | grep -oP 'time = \K[0-9.]+')
    if [ -n "$serial_time" ] && [ -n "$simd_time" ]; then
        speedup=$(echo "$serial_time / $simd_time" | bc -l)
        printf "%d, %.6f, %.6f, %.2f\n" $n $serial_time $simd_time $speedup >> results.csv
        echo "done (speedup $speedup)"
    else
        echo "failed to get times"
    fi
done
echo "=== Results ==="
cat results.csv
