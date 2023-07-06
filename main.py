import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--if_confidence', help='Confidence flag in the environment')

args = parser.parse_args()

# Run the script.py file with arguments
subprocess.run(['python', 'lstm_bl.py', '--if_confidence', args.if_confidence])
subprocess.run(['python', 'transformer_bl.py', '--if_confidence', args.if_confidence])
subprocess.run(['python', 'mlp_bl.py', '--if_confidence', args.if_confidence])
subprocess.run(['python', 'lstm_bl.py', '--if_confidence', args.if_confidence])
subprocess.run(['python', 'transformer_bl.py', '--if_confidence', args.if_confidence])
