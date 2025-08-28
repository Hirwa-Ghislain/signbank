#!/usr/bin/env python3
"""
Quick Start Training Script for Sign Language to Text Model
===========================================================

This script provides a simplified way to start training the sign language model
with minimal configuration. It's designed for quick experimentation and testing.

Usage:
    python train_model.py

Author: AI Assistant
Date: 2024
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Import our model classes
from sign_language_to_text_model import (
    SignLanguageDataset, 
    SignLanguageTransformer, 
    SignLanguageTrainer,
    create_model_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are available."""
    logger.info("Checking dependencies...")
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        logger.error("PyTorch not found. Please install: pip install torch")
        return False
    
    # Check transformers
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("Transformers not found. Please install: pip install transformers")
        return False
    
    # Check pandas
    try:
        import pandas
        logger.info(f"Pandas version: {pandas.__version__}")
    except ImportError:
        logger.error("Pandas not found. Please install: pip install pandas")
        return False
    
    # Check SignWriting (optional)
    try:
        from signwriting.tokenizer import SignWritingTokenizer
        logger.info("SignWriting library found")
    except ImportError:
        logger.warning("SignWriting library not found. Install with: pip install git+https://github.com/sign-language-processing/signwriting")
    
    return True


def check_data():
    """Check if the required data files are available."""
    logger.info("Checking data files...")
    
    data_path = Path("data")
    if not data_path.exists():
        logger.error("Data directory not found. Please ensure you have the SignBank+ dataset.")
        return False
    
    # Check for at least one data file
    data_files = [
        "gpt-3.5-expanded.en.csv"
    ]
    
    available_files = []
    for file in data_files:
        if (data_path / file).exists():
            available_files.append(file)
            logger.info(f"Found data file: {file}")
    
    if not available_files:
        logger.error("No data files found. Please ensure you have the SignBank+ dataset.")
        return False
    
    logger.info(f"Found {len(available_files)} data files")
    return True


def create_output_directories():
    """Create necessary output directories."""
    logger.info("Creating output directories...")
    
    directories = ["models", "logs", "checkpoints"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")


def train_model():
    """Main training function."""
    logger.info("Starting Sign Language to Text Model Training")
    
    # Check dependencies and data
    if not check_dependencies():
        logger.error("Dependency check failed. Please install required packages.")
        return
    
    if not check_data():
        logger.error("Data check failed. Please ensure you have the SignBank+ dataset.")
        return
    
    # Create output directories
    create_output_directories()
    
    # Configuration
    config = create_model_config()
    
    # Adjust config for quick training
    config['epochs'] = 5  # Reduce epochs for quick training
    config['batch_size'] = 8  # Smaller batch size for memory efficiency
    
    logger.info(f"Training configuration: {config}")
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    try:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("T5 tokenizer loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load T5 tokenizer: {e}")
        logger.info("Using simple tokenizer")
        from sign_language_to_text_model import SimpleTokenizer
        tokenizer = SimpleTokenizer()
    
    # Load datasets
    logger.info("Loading datasets...")
    try:
        train_dataset = SignLanguageDataset(
            data_path="data", 
            tokenizer=tokenizer, 
            split='train',
            max_length=256  # Reduced for faster training
        )
        
        # Create a smaller validation set for quick training
        dev_dataset = SignLanguageDataset(
            data_path="data", 
            tokenizer=tokenizer, 
            split='dev',
            max_length=256
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Dev dataset size: {len(dev_dataset)}")
        
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    logger.info("Initializing model...")
    try:
        model = SignLanguageTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            max_length=config['max_length'],
            dropout=config['dropout']
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    try:
        trainer = SignLanguageTrainer(
            model=model,
            tokenizer=tokenizer,
            device=config['device'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size']
        )
    except Exception as e:
        logger.error(f"Error initializing trainer: {e}")
        return
    
    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')
    
    try:
        for epoch in range(config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{config['epochs']}")
            
            # Train
            train_loss = trainer.train_epoch(train_loader)
            logger.info(f"Training Loss: {train_loss:.4f}")
            
            # Evaluate
            eval_results = trainer.evaluate(dev_loader)
            logger.info(f"Validation Loss: {eval_results['loss']:.4f}")
            logger.info(f"Validation BLEU: {eval_results['bleu']:.4f}")
            
            # Save checkpoint
            checkpoint_path = f"checkpoints/epoch_{epoch+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': eval_results['loss'],
                'val_bleu': eval_results['bleu']
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if eval_results['loss'] < best_loss:
                best_loss = eval_results['loss']
                best_model_path = "models/best_model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'loss': best_loss,
                    'bleu': eval_results['bleu']
                }, best_model_path)
                logger.info(f"Saved best model: {best_model_path}")
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {best_loss:.4f}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return
    
    # Final evaluation
    logger.info("Running final evaluation...")
    try:
        final_results = trainer.evaluate(dev_loader)
        logger.info(f"Final Validation Loss: {final_results['loss']:.4f}")
        logger.info(f"Final Validation BLEU: {final_results['bleu']:.4f}")
    except Exception as e:
        logger.error(f"Error in final evaluation: {e}")


def main():
    """Main function."""
    print("=" * 60)
    print("Sign Language to Text Model - Quick Start Training")
    print("=" * 60)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Start training
    train_model()
    
    print("=" * 60)
    print("Training script completed!")
    print("Check the 'models/' directory for your trained model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
