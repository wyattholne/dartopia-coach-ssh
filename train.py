import os
import argparse
from ultralytics import YOLO
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(args):
    try:
        # Print environment for debugging
        logger.info("Environment variables:")
        for key, value in os.environ.items():
            if key.startswith('SM_'):
                logger.info(f"{key}: {value}")

        # Load model
        logger.info(f"Loading model from: {args.model}")
        model = YOLO(args.model)

        # Get data path
        data_path = os.path.join(os.environ['SM_CHANNEL_TRAINING'], 'data.yaml')
        logger.info(f"Using data from: {data_path}")

        # Determine device
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        # Train
        # Get the trainer object
        trainer = model.trainer
        # Add a callback to the on_train_epoch_end event
        trainer.add_callback('on_train_epoch_end', on_train_epoch_end)

        model.train(
            data=data_path,
            epochs=args.epochs,
            device=device,  # Use GPU if available
            lr0=args.lr0,
            batch=8,
            imgsz=416
        )

        # Save model
        output_path = os.path.join(os.environ['SM_MODEL_DIR'], 'best.pt')
        model.save(output_path)
        logger.info(f"Saved model to: {output_path}")

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise

def on_train_epoch_end(trainer):
    # Log the training and validation loss at the end of each epoch
    epoch = trainer.epoch
    train_loss = trainer.loss
    val_loss = trainer.validator.loss
    logger.info(f"Epoch {epoch + 1} Train Loss: {train_loss}")
    logger.info(f"Epoch {epoch + 1} Val Loss: {val_loss}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr0', type=float, default=0.01)

    args = parser.parse_args()
    train(args)
