import argparse
import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import timeit
import statistics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type = int, default = 12)
    parser.add_argument("--num_steps", type = int, default = 10)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], default="train_step")
    return parser.parse_args()

def forward(model, x):
    logits = model(x)
    return logits

def forward_backwards(model, x):
    logits = model(x)
    loss = cross_entropy(logits, x)
    loss.backward()
    model.zero_grad()
    return loss

def train_step(model, x, optimizer):
    optimizer.zero_grad()
    logits = model(x)
    loss = cross_entropy(logits, x)
    loss.backward()
    optimizer.step()
    return loss

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
    optimizer = AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, vocab_size, (batch_size, context_len), device = device)
    for step in range(5):
        if args.mode == "forward":
            forward(model, x)
        elif args.mode == "forward_backward":
            forward_backwards(model, x)
        elif args.mode == "train_step":
            train_step(model, x, optimizer)
    torch.cuda.synchronize()
    start = timeit.default_timer()
    results = []
    for step in range(args.num_steps):
        step_start = timeit.default_timer()
        if args.mode == "forward":
            forward(model, x)
        elif args.mode == "forward_backward":
            forward_backwards(model, x)
        elif args.mode == "train_step":
            train_step(model, x, optimizer)
        torch.cuda.synchronize()
        step_elapsed = timeit.default_timer() - step_start
        results.append(step_elapsed)
    elapsed = timeit.default_timer() - start
    print(f"all steps combined took {elapsed} seconds")
    average_step_time = statistics.mean(results)
    results_std = statistics.stdev(results)
    print(f"average step time: {average_step_time}")
    print(f"step time standard deviation: {results_std}")

