import subprocess
import argparse
from mlp_bl import run_mlp_bl
from lstm_bl import run_lstm_bl
from transformer_bl import run_transformer_bl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--if_confidence', help='Confidence flag in the environment',type=str)
    parser.add_argument('--model', help='Model to run',type=str)

    args = parser.parse_args()

    if args.model == "mlp":
        run_mlp_bl(args.if_confidence)
    elif args.model=="lstm":
        run_lstm_bl(args.if_confidence)
    elif args.model=="trans":
        run_transformer_bl(args.if_confidence)
