#!/bin/bash
# M7/M6 GCP session — VM bootstrap (runs ON the VM after first ssh).
# VM assumptions: Deep Learning VM pytorch image, 1x T4/L4, CUDA ready.
set -e
echo "== deps =="
pip install -q datasets flash-linear-attention 2>&1 | tail -1 || \
  pip install -q datasets  # FLA optional; needs triton/CUDA to import
echo "== layout =="
mkdir -p ~/morpho ~/.genomic_benchmarks
echo "== gpu check =="
python3 - << 'EOF'
import torch
print('cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
EOF
echo "VM ready. Now rsync repo + data from the Mac, then run chains."
