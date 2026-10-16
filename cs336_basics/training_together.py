import numpy as np
import torch
import cs336_basics.transformer as transformer
import cs336_basics.training as training
import argparse
""" Training loopWrite a script that runs a training loop to train your model on user-provided input. In particular, we recommend that your training script allow for (at least) the following:
1. Ability to configure and control the various model and optimizer hyperparameters.
2.Memory-efficient loading of large training and validation datasets with np.memmap.
3.Serializing checkpoints to a user-provided path.
4.Periodically logging training and validation performance (e.g., to console and/or an external service like Weights and Biases)."""

def train_loop(model, optimizer, dataset, batch_size, context_length, device, epochs, log_interval, checkpoint_interval, checkpoint_path):
    for epoch in range(epochs):
        num_batches = len(dataset) // batch_size    
        for _ in range(num_batches):
            input, target = training.get_batch(dataset, batch_size, context_length, device)
            optimizer.zero_grad()
            loss = model(input, target).mean()
            loss.backward()
            optimizer.step()
            if epoch % log_interval == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
            if epoch % checkpoint_interval == 0:
                training.save_checkpoint(model, optimizer, epoch, checkpoint_path)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_path", type=str, default="dataset.bin")
    args = parser.parse_args()

    # Create model
    model = transformer.TransformerLM(args.vocab_size, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.context_length, args.rope_theta)
    optimizer = training.AdamW(model.parameters(), lr=args.lr)
    dataset = np.memmap(args.dataset_path, dtype=np.uint16, mode="r")
    train_loop(model, optimizer, dataset, args.batch_size, args.context_length, args.device, args.epochs, args.log_interval, args.checkpoint_interval, args.checkpoint_path)