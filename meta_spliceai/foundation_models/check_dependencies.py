#!/usr/bin/env python3
"""
Utility script to check dependencies for the foundation_model package.
Run this script to verify your installation before using the models.
"""

from meta_spliceai.foundation_model.verify_install import verify_installation, run_quick_test

if __name__ == "__main__":
    print("Checking dependencies for meta-spliceai foundation_model...")
    
    # First verify installation of required packages
    all_required, results = verify_installation(include_optional=True, verbose=True)
    
    # If all required packages are installed, run quick tests
    if all_required:
        print("\nAll required dependencies are installed. Running functionality tests...")
        all_tests_passed = run_quick_test()
        
        if all_tests_passed:
            print("\n✅ Your environment is correctly set up for meta-spliceai foundation models!")
        else:
            print("\n⚠️ Some functionality tests failed. The models might not work as expected.")
    else:
        print("\n❌ Some required dependencies are missing. Please install them before using this package.")
