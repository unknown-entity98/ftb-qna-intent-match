
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
    
    print("\n🎉 All imports successful!")
    print("Ready to run intent classification system")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Run fix_dependencies.py first")
    sys.exit(1)
