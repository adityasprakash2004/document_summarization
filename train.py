import argparse
from dataprep import get_tokenized_datasets
from optimize import fine_tune

def parse_args():
    parser = argparse.ArgumentParser(description="Train document summarization model")
    parser.add_argument(
        "--model_name_or_path", type=str, default="facebook/bart-base",
        help="Pre-trained model identifier from Hugging Face or local path."
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory where the final model and checkpoints will be saved."
    )
    parser.add_argument(
        "--max_input_length", type=int, default=1024,
        help="Maximum input sequence length for the encoder."
    )
    parser.add_argument(
        "--max_target_length", type=int, default=128,
        help="Maximum output sequence length for the decoder."
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4,
        help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=4,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5,
        help="Initial learning rate."
    )
    parser.add_argument(
        "--logging_steps", type=int, default=500,
        help="Log training metrics every X steps."
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000,
        help="Save a checkpoint every X steps."
    )
    parser.add_argument(
        "--eval_steps", type=int, default=1000,
        help="Run evaluation every X steps."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    datasets, tokenizer = get_tokenized_datasets(
        model_name_or_path=args.model_name_or_path,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size
    )

    fine_tune(
        model_name_or_path=args.model_name_or_path,
        datasets=datasets,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps
    )

if __name__ == "__main__":
    main()
