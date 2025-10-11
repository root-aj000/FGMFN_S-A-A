import argparse
import subprocess
import sys
import os

def run_cmd(cmd_list):
    """Run list of shell commands sequentially."""
    for cmd in cmd_list:
        print(f"\n>>> Running: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"❌ Command failed: {cmd}")
            sys.exit(1)

def install_env():
    return [
        "pip install -r requirements.txt",
        "echo '✅ Environment setup complete!'"
    ]

def prepare_data():
    return [
        "python -m dataset_generator.prepare_data",
        "python dataset_generator/preprocess.py"
    ]

def train_model():
    return [
        "python -m training.train",
        "echo '✅ Model saved in models/saved_models/'"
    ]

def evaluate_model():
    return [
        "python -m training.evaluate",
        "echo '📊 Evaluation report saved in data/logs/'"
    ]

def serve_api():
    return [
        "python -m server.app"
    ]
 
def launch_frontend():
    return [
        "python -m http.server 5500",
        "echo '🌐 Frontend launched in browser'"
    ]

def main():
    parser = argparse.ArgumentParser(description="Project Setup & Workflow Runner")

    parser.add_argument("--install", action="store_true", help="Install requirements and prepare environment")
    parser.add_argument("--prepare-data", action="store_true", help="Generate & preprocess dataset")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model")
    parser.add_argument("--serve", action="store_true", help="Run FastAPI server")
    parser.add_argument("--frontend", action="store_true", help="Launch frontend in browser")

    args = parser.parse_args()

    # If no arguments → run everything sequentially
    if not any(vars(args).values()):
        print("🚀 No arguments provided → Running FULL pipeline...\n")
        full_pipeline = (
            install_env()
            + prepare_data()
            + train_model()
            + evaluate_model()
            + serve_api()
            + launch_frontend()
        )
        run_cmd(full_pipeline)
        return

    # Otherwise run selected steps
    if args.install:
        run_cmd(install_env())
    if args.prepare_data:
        run_cmd(prepare_data())
    if args.train:
        run_cmd(train_model())
    if args.evaluate:
        run_cmd(evaluate_model())
    if args.serve:
        run_cmd(serve_api())
    if args.frontend:
        run_cmd(launch_frontend())

if __name__ == "__main__":
    main()