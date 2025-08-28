#!/usr/bin/env python3
"""
Sign Language to Text Model Demo
================================

A simple demo script to showcase the sign language to text translation model.
This script provides examples and quick testing capabilities.

Usage:
    python demo.py

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import torch
from pathlib import Path

# Import our model classes
from sign_language_to_text_model import SignLanguageTransformer, SignLanguageDataset
from inference import SignLanguageInference

def print_banner():
    """Print a nice banner for the demo."""
    print("=" * 70)
    print("🎯 Sign Language to Text Model Demo")
    print("=" * 70)
    print("This demo showcases the sign language translation capabilities.")
    print("The model converts SignWriting notation to English text.")
    print("=" * 70)

def check_model_exists():
    """Check if a trained model exists."""
    model_path = Path("models/best_model.pth")
    if model_path.exists():
        print(f"✅ Found trained model: {model_path}")
        return True
    else:
        print(f"❌ No trained model found at: {model_path}")
        print("   Please run training first: python train_model.py")
        return False

def load_sample_signs():
    """Load sample SignWriting signs for demonstration."""
    sample_signs = {
        "Alphabet": {
            "A": "M510x508S1f720490x493",
            "B": "M511x507S14702489x493", 
            "C": "M509x510S16d20492x490",
            "D": "M508x515S10120492x485",
            "E": "M512x509S16710489x491",
            "F": "M515x514S1d220486x486",
            "G": "M515x508S10002485x493",
            "H": "M518x533S33b00482x483S11e10465x503S20500495x520S26500466x488",
            "I": "M511x510S19220490x491",
            "J": "M517x517S19220496x498S2e706484x484"
        },
        "Common Words": {
            "Hello": "M536x571S15a01494x528S15a0a466x559S2e805488x542S2ff00482x482S15a21467x510S2ed02513x551",
            "Thank You": "M520x526S15a10480x475S1fa1e502x479S2df03501x502",
            "Please": "M515x534S15a18485x489S10010498x467S2e700500x501S2f900500x529",
            "Yes": "M520x526S15a10480x475S1fa1e502x479S2df03501x502",
            "No": "M515x534S15a18485x489S10010498x467S2e700500x501S2f900500x529"
        },
        "Numbers": {
            "1": "M508x515S10020493x485",
            "2": "M508x515S10e20493x485", 
            "3": "M509x515S18620491x485",
            "4": "M511x516S14420489x485",
            "5": "M512x516S14c20489x485"
        }
    }
    return sample_signs

def demo_without_model():
    """Demo without a trained model - show sample data."""
    print("\n📊 Demo Mode (No Trained Model)")
    print("-" * 40)
    
    sample_signs = load_sample_signs()
    
    print("Sample SignWriting notations:")
    print()
    
    for category, signs in sample_signs.items():
        print(f"🔤 {category}:")
        for word, sign in signs.items():
            print(f"   {word:10} → {sign}")
        print()
    
    print("💡 To see actual translations, train a model first:")
    print("   python train_model.py")

def demo_with_model():
    """Demo with a trained model."""
    print("\n🤖 Demo Mode (With Trained Model)")
    print("-" * 40)
    
    try:
        # Initialize inference
        inference = SignLanguageInference('models/best_model.pth')
        print("✅ Model loaded successfully!")
        
        # Load sample signs
        sample_signs = load_sample_signs()
        
        # Test translations
        print("\n🔤 Testing Alphabet Signs:")
        print("-" * 30)
        
        for letter, sign in sample_signs["Alphabet"].items():
            try:
                result = inference.translate(sign)
                print(f"{letter:2} → {result['translation']:15} (confidence: {result['confidence']:.3f})")
            except Exception as e:
                print(f"{letter:2} → Error: {str(e)}")
        
        print("\n📝 Testing Common Words:")
        print("-" * 30)
        
        for word, sign in sample_signs["Common Words"].items():
            try:
                result = inference.translate(sign)
                print(f"{word:10} → {result['translation']:20} (confidence: {result['confidence']:.3f})")
            except Exception as e:
                print(f"{word:10} → Error: {str(e)}")
        
        # Interactive mode option
        print("\n🎮 Interactive Mode:")
        print("Would you like to try interactive mode? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes']:
                inference.interactive_mode()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   The model might not be properly trained or saved.")

def show_data_statistics():
    """Show statistics about the available data."""
    print("\n📈 Data Statistics")
    print("-" * 20)
    
    data_path = Path("data")
    if not data_path.exists():
        print("❌ No data directory found")
        return
    
    data_files = [
        "gpt-3.5-expanded.en.csv",
        "gpt-3.5-cleaned.csv", 
        "manually-cleaned.csv",
        "bible.csv",
        "fingerspelling.csv"
    ]
    
    total_entries = 0
    
    for file in data_files:
        file_path = data_path / file
        if file_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                entries = len(df)
                total_entries += entries
                print(f"📄 {file:25} → {entries:,} entries")
            except Exception as e:
                print(f"📄 {file:25} → Error reading file")
        else:
            print(f"📄 {file:25} → Not found")
    
    print(f"\n📊 Total entries: {total_entries:,}")

def show_model_info():
    """Show information about the model architecture."""
    print("\n🏗️  Model Architecture")
    print("-" * 20)
    
    config = {
        'model_type': 'transformer',
        'vocab_size': 32000,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 2048,
        'max_length': 512,
        'dropout': 0.1
    }
    
    print("Architecture: Transformer")
    print(f"Vocabulary Size: {config['vocab_size']:,}")
    print(f"Model Dimension: {config['d_model']}")
    print(f"Attention Heads: {config['nhead']}")
    print(f"Layers: {config['num_layers']}")
    print(f"Feedforward Dim: {config['dim_feedforward']}")
    print(f"Max Length: {config['max_length']}")
    print(f"Dropout: {config['dropout']}")

def show_signwriting_info():
    """Show information about SignWriting notation."""
    print("\n📝 SignWriting Notation")
    print("-" * 20)
    
    print("SignWriting is a written form of sign language that uses symbols to represent:")
    print("• Hand shapes and movements")
    print("• Facial expressions") 
    print("• Body positions")
    print("• Spatial relationships")
    print()
    print("Example: M510x508S1f720490x493")
    print("• M: Manual symbol")
    print("• 510x508: Position coordinates")
    print("• S1f720490x493: Hand shape and orientation")
    print()
    print("💡 The model learns to translate these symbols into English text.")

def main():
    """Main demo function."""
    print_banner()
    
    # Check if model exists
    model_exists = check_model_exists()
    
    # Show data statistics
    show_data_statistics()
    
    # Show model architecture
    show_model_info()
    
    # Show SignWriting info
    show_signwriting_info()
    
    # Run appropriate demo
    if model_exists:
        demo_with_model()
    else:
        demo_without_model()
    
    print("\n" + "=" * 70)
    print("🎉 Demo completed!")
    print("=" * 70)
    print("For more information, see README_SIGN_LANGUAGE_MODEL.md")
    print("=" * 70)

if __name__ == "__main__":
    main()
