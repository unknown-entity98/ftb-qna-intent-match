#!/usr/bin/env python3
"""
Fix dependency conflicts and install packages properly
Run this first if you're having dependency issues
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error during {description}: {e}")
        return False

def fix_dependencies():
    """Fix dependency conflicts"""
    print("🛠️  Fixing MCP Intent Classification Dependencies")
    print("=" * 50)
    
    # Step 1: Create virtual environment (recommended)
    print("\n💡 RECOMMENDATION: Use a virtual environment")
    print("To create one: python -m venv intent_env")
    print("To activate: source intent_env/bin/activate (Linux/Mac) or intent_env\\Scripts\\activate (Windows)")
    print("\nContinuing with current environment...")
    
    # Step 2: Upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Step 3: Install core packages first
    core_packages = [
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0", 
        "python-dotenv>=1.0.0"
    ]
    
    for package in core_packages:
        run_command(f"{sys.executable} -m pip install '{package}'", f"Installing {package}")
    
    # Step 4: Install MCP (this might conflict with langchain)
    print("\n⚠️  Installing MCP (may conflict with existing langchain)")
    print("If you need langchain, consider using separate environments")
    
    run_command(f"{sys.executable} -m pip install mcp", "Installing MCP")
    
    # Step 5: Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("mcp", "MCP"),
        ("dotenv", "Python-dotenv")
    ]
    
    all_good = True
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} import successful")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            all_good = False
    
    if all_good:
        print("\n🎉 All dependencies installed successfully!")
        print("\nNext steps:")
        print("1. Run: python intent_client_fixed.py --test")
        print("2. If test passes, run: python intent_client_fixed.py --demo")
    else:
        print("\n⚠️  Some dependencies failed to install")
        print("Consider using a virtual environment or resolving conflicts manually")
    
    return all_good

def create_minimal_test():
    """Create a minimal test to verify everything works"""
    
    test_code = '''
import sys
print("🧪 Minimal MCP Test")
print("=" * 30)

try:
    import numpy as np
    print("✅ NumPy imported")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("✅ Scikit-learn imported")
    
    from mcp.server import Server
    print("✅ MCP server imported")
    
    from mcp import ClientSession
    print("✅ MCP client imported")
    
    print("\\n🎉 All imports successful!")
    print("Ready to run intent classification system")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Run fix_dependencies.py first")
    sys.exit(1)
'''
    
    with open("test_imports.py", "w") as f:
        f.write(test_code)
    
    print("\n📝 Created test_imports.py")
    print("Run: python test_imports.py to verify installation")

def main():
    """Main function"""
    print("🚀 MCP Intent Classification Dependency Fixer")
    print("This will help resolve dependency conflicts")
    print()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Fix dependencies")
    parser.add_argument("--test", action="store_true", help="Create test file")
    parser.add_argument("--venv", action="store_true", help="Show virtual environment setup")
    
    args = parser.parse_args()
    
    if args.venv:
        print("📦 Virtual Environment Setup (Recommended)")
        print("=" * 45)
        print("1. Create virtual environment:")
        print("   python -m venv intent_classification_env")
        print()
        print("2. Activate it:")
        print("   # On Windows:")
        print("   intent_classification_env\\Scripts\\activate")
        print()
        print("   # On Linux/Mac:")
        print("   source intent_classification_env/bin/activate")
        print()
        print("3. Install packages:")
        print("   python fix_dependencies.py --fix")
        print()
        print("4. Run the system:")
        print("   python intent_client_fixed.py --demo")
        
    elif args.test:
        create_minimal_test()
        
    elif args.fix:
        fix_dependencies()
        create_minimal_test()
        
    else:
        print("Options:")
        print("  --fix   : Fix dependency conflicts")
        print("  --test  : Create import test file") 
        print("  --venv  : Show virtual environment setup")
        print()
        print("Recommended: python fix_dependencies.py --fix")

if __name__ == "__main__":
    main()
