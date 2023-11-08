import subprocess
import argparse
from mlp_bl_sp import run_mlp_bl_sp
from mlp_bl_rusell import run_mlp_bl_rusell
from lstm_bl_sp import run_lstm_bl_sp
from lstm_bl_rusell import run_lstm_bl_rusell
from transformer_bl_sp import run_transformer_bl_sp
from transformer_bl_rusell import run_transformer_bl_rusell

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--if_confidence', help='Confidence flag in the environment',type=str)
    parser.add_argument('--model', help='Model to run',type=str)
    parser.add_argument('--stock',help='Which stock to use',type=str)
    args = parser.parse_args()

    if args.model == "mlp" and args.stock=="sp":
        run_mlp_bl_sp(args.if_confidence)
    elif args.model=="mlp" and args.stock=="rusell":
        run_mlp_bl_rusell(args.if_confidence)
    elif args.model=="lstm":
        run_lstm_bl_sp(args.if_confidence)
        run_lstm_bl_rusell(args.if_confidence)
    elif args.model=="trans":
        run_transformer_bl_sp(args.if_confidence)
        run_transformer_bl_rusell(args.if_confidence)
