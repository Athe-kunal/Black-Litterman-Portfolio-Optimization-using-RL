#!/bin/bash
echo >> Started with confidence
python3 mlp_bl.py
wait
python3 lstm_bl.py
wait
python3 transformer_bl.py
wait

echo >> Started without confidence
python3 mlp_bl.py -if False
wait
python3 lstm_bl.py -if False
wait
python3 transformer_bl.py -if False
