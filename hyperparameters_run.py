import subprocess
import json
from datetime import datetime
import argparse


def run_training(learning_rate, batch_size, output_p):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_p}/output_{timestamp}_lr{learning_rate}_bs{batch_size}"

    print(f"Running training with LR={learning_rate}, Batch Size={batch_size}...")

    command = [
        "python",
        "train.py",  # Replace with the actual name of your training script
        "--input-channels",
        "3",  # Modify these with your desired parameters
        "--base-path",
        "/kaggle/working/defect_detection/KSDD2/train",
        "--dataset-json-path",
        "/kaggle/working/defect_detection/KSDD2/test",
        "--train-percentage",
        "1.0",
        "--height",
        "640",
        "--width",
        "232",
        "--dil-ksize",
        "15",
        "--mixed-sup-N",
        "246",
        "--dist-trans-w",
        "3.0",
        "--dist-trans-p",
        "2.0",
        "--shuffle-buf-size",
        "500",
        "--batch-size",
        str(batch_size),
        # "--optimizer", "adam",
        "--learning-rate",
        str(learning_rate),
        # "--learning-decay", "False",
        "--epochs",
        "5",
        "--delta",
        "1.0",
        "--log-interval",
        "5",
        # "--metrics", "accuracy",  # Add more metrics if needed
        "--test-on-cpu",
        "False",
        "--output-path",
        output_path,
        # "--networks-config-path", "",
        # "--aug-dir", "/path/to/augmented_images",  # Optional, add if needed
        # "--aug-trained-model", "/path/to/augmented_model",  # Optional, add if needed
    ]

    subprocess.run(command)

    # Read metrics from the output file
    with open(f"{output_path}/metrics.json", "r") as f:
        print("Reading metrics...")
        metrics = json.load(f)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run training script with different hyperparameters.')
    parser.add_argument('--output-path', required=True, type=str, help='Path to save results')
    args = parser.parse_args()

    # Define hyperparameter combinations to try
    learning_rates = [0.0001, 0.001, 0.01]
    batch_sizes = [1, 16, 32]

    # Run training with different hyperparameters
    for lr in learning_rates:
        for bs in batch_sizes:
            metrics_result = run_training(lr, bs, args.output_path)
            print(f"Metrics for LR={lr}, Batch Size={bs}: {metrics_result}")

if __name__ == '__main__':
    main()
