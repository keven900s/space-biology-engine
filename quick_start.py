#!/usr/bin/env python3
"""
Quick Start Script for NASA Space Biology Knowledge Engine
Run this to set up everything for the hackathon demo

Usage:
    python quick_start.py --mode demo    # Process 50 papers (fast)
    python quick_start.py --mode full    # Process all 608 papers (slow)
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess

def check_python_version():
    """Ensure Python 3.9+"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    dirs = ['data', 'data/processed', 'src', 'utils', 'tests', 'docs']
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {dir_name}/")
    print("✅ Directories created")

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path('requirements.txt').exists():
        print("⚠️  requirements.txt not found. Creating it...")
        create_requirements_file()
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("   Try manually: pip install -r requirements.txt")
        sys.exit(1)

def create_requirements_file():
    """Create requirements.txt if it doesn't exist"""
    requirements = """streamlit>=1.30.0
pandas>=2.1.0
numpy>=1.24.0
plotly>=5.17.0
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
PyPDF2>=3.0.0
transformers>=4.35.0
sentence-transformers>=2.2.0
torch>=2.1.0
spacy>=3.7.0
scikit-learn>=1.3.0
networkx>=3.2.0
pyvis>=0.3.2
tqdm>=4.66.0
python-dotenv>=1.0.0
"""
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("   Created requirements.txt")

def download_models():
    """Download required NLP models"""
    print("\n🤖 Downloading NLP models...")
    try:
        subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], 
                      check=True, capture_output=True)
        print("✅ Spacy model downloaded")
    except subprocess.CalledProcessError:
        print("⚠️  Could not download spacy model automatically")
        print("   Run manually: python -m spacy download en_core_web_sm")

def check_dataset():
    """Check if dataset exists"""
    print("\n📊 Checking for dataset...")
    csv_path = Path('data/SB_publications_PMC.csv')
    
    if not csv_path.exists():
        print("⚠️  Dataset not found!")
        print("   Please place 'SB_publications_PMC.csv' in the data/ folder")
        print("   Download from: https://github.com/nasa/..." )
        return False
    
    # Check file size
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Dataset found: {len(df)} papers")
        return True
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False

def create_demo_data(mode='demo'):
    """Process papers for demo"""
    print(f"\n⚙️  Processing papers ({mode} mode)...")
    
    try:
        import pandas as pd
        from src.data_ingestion import DataIngestion
        from src.nlp_processor import NLPProcessor
        from config import Config
        
        config = Config()
        
        # Load CSV
        df = pd.read_csv('data/SB_publications_PMC.csv')
        print(f"   Loaded {len(df)} papers from CSV")
        
        # Determine how many to process
        if mode == 'demo':
            n_papers = min(50, len(df))
            df = df.head(n_papers)
            print(f"   Demo mode: Processing first {n_papers} papers")
        else:
            print(f"   Full mode: Processing all {len(df)} papers (this will take time!)")
        
        # Process papers
        ingestion = DataIngestion(config)
        papers = ingestion.process_all_papers(df, use_cache=True)
        
        # Add NLP features to first batch
        print("   Adding AI summaries and entity extraction...")
        nlp = NLPProcessor(config)
        
        processed_count = 0
        for paper in papers[:min(25, len(papers))]:  # Process first 25 fully
            if paper.get('abstract'):
                try:
                    paper['summary'] = nlp.generate_summary(paper['abstract'])
                    paper['entities'] = nlp.extract_entities(paper['abstract'])
                    processed_count += 1
                    if processed_count % 5 == 0:
                        print(f"   Processed {processed_count} papers...")
                except Exception as e:
                    print(f"   Warning: Could not process paper: {e}")
        
        print(f"✅ Processed {processed_count} papers with AI features")
        print("   Results cached for quick loading")
        return True
        
    except ImportError as e:
        print(f"❌ Missing module: {e}")
        print("   Make sure all project files are in place")
        return False
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_env_file():
    """Create .env file with default settings"""
    print("\n⚙️  Creating configuration...")
    env_content = """# NASA Space Biology Knowledge Engine Configuration

# Processing Settings
MAX_PAPERS=608
CACHE_ENABLED=true

# Model Settings
SUMMARIZER_MODEL=facebook/bart-large-cnn
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional: Add API keys if using external services
# OPENAI_API_KEY=your_key_here
"""
    
    if not Path('.env').exists():
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env configuration file")
    else:
        print("   .env already exists")

def run_tests():
    """Run basic tests"""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        import streamlit
        import pandas
        import plotly
        import transformers
        print("✅ Core libraries imported successfully")
        
        # Test text processor
        from utils.text_processor import TextProcessor
        processor = TextProcessor()
        test_text = "Testing the NASA Space Biology Engine"
        cleaned = processor.clean_text(test_text)
        print("✅ Text processor working")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_next_steps():
    """Print what to do next"""
    print("\n" + "="*60)
    print("🎉 Setup Complete! Next Steps:")
    print("="*60)
    print("\n1️⃣  Start the application:")
    print("   streamlit run app.py")
    print("\n2️⃣  Open your browser to:")
    print("   http://localhost:8501")
    print("\n3️⃣  Explore the features:")
    print("   - 🔍 Semantic Search")
    print("   - 📊 Analytics Dashboard")
    print("   - 🕸️  Knowledge Graph")
    print("   - 🎯 Mission Insights")
    print("\n4️⃣  For hackathon:")
    print("   - Prepare your demo script")
    print("   - Take screenshots")
    print("   - Record a video demo")
    print("\n💡 Tips:")
    print("   - Use 'demo' mode (50 papers) for fast iteration")
    print("   - Cache is enabled for quick restarts")
    print("   - Check docs/ for more information")
    print("\n📚 Documentation:")
    print("   - README.md - Project overview")
    print("   - docs/SETUP.md - Detailed setup guide")
    print("   - docs/API.md - API documentation")
    print("\n🐛 Having issues?")
    print("   - Check requirements.txt is complete")
    print("   - Ensure Python 3.9+ is installed")
    print("   - Verify dataset is in data/ folder")
    print("\n🚀 Good luck with the hackathon!")
    print("="*60 + "\n")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description='Quick setup for NASA Space Biology Knowledge Engine'
    )
    parser.add_argument(
        '--mode',
        choices=['demo', 'full', 'minimal'],
        default='demo',
        help='Setup mode: demo (50 papers), full (all 608), minimal (no processing)'
    )
    parser.add_argument(
        '--skip-models',
        action='store_true',
        help='Skip downloading NLP models'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 NASA SPACE BIOLOGY KNOWLEDGE ENGINE")
    print("   Quick Start Setup Script")
    print("="*60)
    
    # Step-by-step setup
    check_python_version()
    setup_directories()
    install_dependencies()
    
    if not args.skip_models:
        download_models()
    
    create_env_file()
    
    dataset_exists = check_dataset()
    
    if dataset_exists and args.mode != 'minimal':
        success = create_demo_data(args.mode)
        if not success:
            print("\n⚠️  Data processing failed, but you can still run the app")
    
    # Run tests
    tests_passed = run_tests()
    
    # Summary
    print("\n" + "="*60)
    print("📋 Setup Summary")
    print("="*60)
    print(f"✅ Directories created")
    print(f"✅ Dependencies installed")
    print(f"{'✅' if not args.skip_models else '⏭️ '} NLP models {'downloaded' if not args.skip_models else 'skipped'}")
    print(f"{'✅' if dataset_exists else '⚠️ '} Dataset {'found' if dataset_exists else 'not found'}")
    print(f"{'✅' if args.mode != 'minimal' else '⏭️ '} Data processing {'complete' if args.mode != 'minimal' else 'skipped'}")
    print(f"{'✅' if tests_passed else '⚠️ '} Tests {'passed' if tests_passed else 'had issues'}")
    
    if dataset_exists and tests_passed:
        print_next_steps()
    else:
        print("\n⚠️  Some issues detected. Please check the messages above.")
        if not dataset_exists:
            print("\n📥 To get the dataset:")
            print("   1. Download SB_publications_PMC.csv from the NASA repository")
            print("   2. Place it in the data/ folder")
            print("   3. Run this script again")

if __name__ == "__main__":
    main()