#!/usr/bin/env bash

N_epochs=200
for t in {1..32}; do
    python training.py \
        --N=2 \
        --latsize="(64,64)" \
        --batch_size=100 \
        --i=0 \
        --j=1 \
        --delta="($t,0)" \
        --source=Tej \
        --run \
        --epochs=$N_epochs
done
