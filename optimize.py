from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch

def fine_tune(model_name_or_path, datasets, tokenizer, output_dir, epochs, train_batch_size, 
              eval_batch_size, learning_rate, logging_steps, save_steps, eval_steps):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_total_limit=2,
        learning_rate=learning_rate,
        remove_unused_columns=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
