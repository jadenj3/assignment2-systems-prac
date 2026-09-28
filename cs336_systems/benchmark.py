import argparse
import torch
from cs336_basics.model import BasicsTransformerLM
import timeit

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type = int, default = 12)
    parser.add_argument("--num_steps", type = int, default = 10)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], default="forward")
    return parser.parse_args()

if __name__== "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_len = 512
    vocab_size = 10000
    batch_size = 4
    print("in main")
    model = BasicsTransformerLM(
        vocab_size = vocab_size,
        context_length = context_len,
        d_model = 768,
        num_layers = args.num_layers,
        num_heads = 12,
        d_ff = 3072,
        rope_theta = 10_000.0
    )
    model.to(device)
    x = torch.randint(0, vocab_size, (batch_size, context_len), device = device)
    for step in range(5):
        model(x)
    start = timeit.default_timer()
    torch.cuda.synchronize()
    for step in range(args.num_steps):
        if args.mode == "forward":
            model(x)
        elif args.mode == "forward_backward":
            pass
        elif args.mode == "train_step":
            pass
        torch.cuda.synchronize()
    elapsed = timeit.default_timer() - start
    print(f"all steps combined took {elapsed} seconds")

