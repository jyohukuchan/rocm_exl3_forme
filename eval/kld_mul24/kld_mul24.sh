cd /src
export PYTHONPATH=/work/lib-mul24-r9700:/src
M=/work/models/qwen38-27b-exl3-4bpw-mul24
Q=/work/kld27b-mul24b
mkdir -p $Q
C=/work/kld27b/cases-v1.json
python3 /work/kld-tools/exl3_dump.py --model $M --manifest $C --output-dir $Q/fp16 > $Q/fp16.log 2>&1
echo "dump_fp16 exit=$?" >> $Q/status.log
EXL3_INT8_PREFILL=1 EXL3_INT8_FIXED_GRID=1 EXL3_INT8_MIN_ROWS=0 python3 /work/kld-tools/exl3_dump.py --model $M --manifest $C --output-dir $Q/int8 > $Q/int8.log 2>&1
echo "dump_int8 exit=$?" >> $Q/status.log
