#!/usr/bin/env python3
"""
Sign Language to Text Inference
===============================

Real-time inference script for converting sign language (SignWriting) to English text.
This script can be used for live sign language translation applications.

Features:
- Load trained models
- Real-time inference
- Batch processing
- Multiple output formats
- Confidence scoring

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime

# Import our model classes
from sign_language_to_text_model import SignLanguageTransformer, SignLanguageDataset

# Transformers
from transformers import T5Tokenizer, AutoTokenizer

# SignWriting
try:
    from signwriting.tokenizer import SignWritingTokenizer
except ImportError:
    print("Warning: SignWriting library not found")
    SignWritingTokenizer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignLanguageInference:
    """
    Inference class for sign language to text translation.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'auto',
                 max_length: int = 512):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model
            device: Device to use ('auto', 'cpu', 'cuda')
            max_length: Maximum sequence length
        """
        self.model_path = Path(model_path)
        self.max_length = max_length
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model and tokenizer
        self.model, self.tokenizer, self.config = self._load_model()
        self.signwriting_tokenizer = SignWritingTokenizer() if SignWritingTokenizer else None
        
        logger.info(f"Inference initialized on device: {self.device}")
    
    def _load_model(self):
        """Load the trained model and configuration."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Initialize tokenizer
        try:
            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            tokenizer.pad_token = tokenizer.eos_token
        except:
            logger.warning("Could not load T5 tokenizer")
            tokenizer = None
        
        # Initialize model
        model = SignLanguageTransformer(
            vocab_size=config.get('vocab_size', 32000),
            d_model=config.get('d_model', 512),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 6),
            dim_feedforward=config.get('dim_feedforward', 2048),
            max_length=config.get('max_length', 512),
            dropout=config.get('dropout', 0.1)
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded successfully. Loss: {checkpoint.get('loss', 'N/A')}")
        
        return model, tokenizer, config
    
    def preprocess_signwriting(self, sign_text: str) -> str:
        """
        Preprocess SignWriting text for inference.
        
        Args:
            sign_text: Raw SignWriting text
            
        Returns:
            Preprocessed SignWriting text
        """
        if self.signwriting_tokenizer:
            # Tokenize SignWriting
            tokens = list(self.signwriting_tokenizer.text_to_tokens(sign_text))
            return " ".join(tokens)
        else:
            # Fallback: return raw text
            return sign_text
    
    def translate(self, 
                 sign_text: str,
                 max_length: Optional[int] = None,
                 num_beams: int = 4,
                 temperature: float = 1.0) -> Dict:
        """
        Translate SignWriting to English text.
        
        Args:
            sign_text: SignWriting text to translate
            max_length: Maximum output length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            
        Returns:
            Dictionary with translation results
        """
        if max_length is None:
            max_length = self.max_length
        
        # Preprocess input
        processed_sign = self.preprocess_signwriting(sign_text)
        
        # Tokenize input
        if self.tokenizer:
            inputs = self.tokenizer(
                processed_sign,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
        else:
            # Fallback: create dummy input
            input_ids = torch.tensor([[0]]).to(self.device)
            attention_mask = torch.tensor([[1]]).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            
            # Get predictions
            pred_ids = torch.argmax(outputs, dim=-1)
            
            # Decode output
            if self.tokenizer:
                translated_text = self.tokenizer.decode(
                    pred_ids[0], 
                    skip_special_tokens=True
                )
            else:
                translated_text = "Translation not available (no tokenizer)"
        
        # Calculate confidence (simplified)
        confidence = self._calculate_confidence(outputs, pred_ids)
        
        return {
            'input': sign_text,
            'translation': translated_text,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_confidence(self, outputs, pred_ids) -> float:
        """Calculate confidence score for the translation."""
        # Get probabilities
        probs = torch.softmax(outputs, dim=-1)
        
        # Get probability of predicted tokens
        batch_size, seq_len, vocab_size = probs.shape
        pred_probs = torch.gather(probs, 2, pred_ids.unsqueeze(-1)).squeeze(-1)
        
        # Calculate average probability (confidence)
        confidence = torch.mean(pred_probs).item()
        
        return confidence
    
    def batch_translate(self, 
                       sign_texts: List[str],
                       max_length: Optional[int] = None) -> List[Dict]:
        """
        Translate multiple SignWriting texts in batch.
        
        Args:
            sign_texts: List of SignWriting texts
            max_length: Maximum output length
            
        Returns:
            List of translation results
        """
        if max_length is None:
            max_length = self.max_length
        
        results = []
        
        for sign_text in sign_texts:
            try:
                result = self.translate(sign_text, max_length)
                results.append(result)
            except Exception as e:
                logger.error(f"Error translating: {e}")
                results.append({
                    'input': sign_text,
                    'translation': f"Error: {str(e)}",
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def interactive_mode(self):
        """Run interactive mode for real-time translation."""
        print("=== Sign Language to Text Translation ===")
        print("Enter SignWriting text (or 'quit' to exit):")
        print("Example: M510x508S1f720490x493")
        
        while True:
            try:
                user_input = input("\nSignWriting: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Translate
                result = self.translate(user_input)
                
                print(f"\nTranslation: {result['translation']}")
                print(f"Confidence: {result['confidence']:.3f}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def load_sample_data():
    """Load sample SignWriting data for testing."""
    sample_signs = [
        "M510x508S1f720490x493",  # A
        "M511x507S14702489x493",  # B
        "M509x510S16d20492x490",  # C
        "M508x515S10120492x485",  # D
        "M512x509S16710489x491",  # E
        "M515x514S1d220486x486",  # F
        "M515x508S10002485x493",  # G
        "M511x510S19220490x491",  # I
        "M515x515S14020486x485",  # K
        "M512x515S1dc20488x485",  # L
    ]
    
    return sample_signs


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sign Language to Text Inference")
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--input', type=str, help='Input SignWriting text')
    parser.add_argument('--file', type=str, help='Input file with SignWriting texts')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--sample', action='store_true',
                       help='Run with sample data')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    try:
        inference = SignLanguageInference(
            model_path=args.model,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize inference: {e}")
        return
    
    # Run different modes
    if args.interactive:
        inference.interactive_mode()
    
    elif args.sample:
        print("Running with sample data...")
        sample_signs = load_sample_data()
        results = inference.batch_translate(sample_signs)
        
        for i, result in enumerate(results):
            print(f"\nSample {i+1}:")
            print(f"Input: {result['input']}")
            print(f"Translation: {result['translation']}")
            print(f"Confidence: {result['confidence']:.3f}")
    
    elif args.file:
        print(f"Processing file: {args.file}")
        try:
            with open(args.file, 'r') as f:
                sign_texts = [line.strip() for line in f if line.strip()]
            
            results = inference.batch_translate(sign_texts)
            
            # Save results
            output_file = f"translations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
    
    elif args.input:
        print(f"Translating: {args.input}")
        result = inference.translate(args.input)
        
        print(f"Translation: {result['translation']}")
        print(f"Confidence: {result['confidence']:.3f}")
    
    else:
        print("No input specified. Use --help for options.")
        print("Running with sample data...")
        sample_signs = load_sample_data()
        results = inference.batch_translate(sample_signs)
        
        for i, result in enumerate(results):
            print(f"\nSample {i+1}:")
            print(f"Input: {result['input']}")
            print(f"Translation: {result['translation']}")
            print(f"Confidence: {result['confidence']:.3f}")


if __name__ == "__main__":
    main()
