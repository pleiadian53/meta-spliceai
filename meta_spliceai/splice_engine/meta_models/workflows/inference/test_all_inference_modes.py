#!/usr/bin/env python3
"""
🧪 Comprehensive Test Suite for All Inference Modes

Tests all three inference modes (hybrid, base_only, meta_only) across 
different gene scenarios to validate proper integration of recalibrated 
meta-model predictions with base-model predictions.

Usage:
    python test_all_inference_modes.py [--dry-run] [--quick]
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

class InferenceModeTestSuite:
    """Comprehensive test suite for inference modes and scenarios."""
    
    def __init__(self, dry_run: bool = False, quick: bool = False):
        self.dry_run = dry_run
        self.quick = quick
        self.base_cmd = [
            "python", "-m", 
            "meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow"
        ]
        self.model_path = "results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl"
        self.training_dataset = "train_pc_1000_3mers"
        self.results = {}
        
        # Test genes for different scenarios
        self.scenario1_genes = ["ENSG00000280739", "ENSG00000095637", "ENSG00000257771"]  # In training
        self.scenario2b_genes = ["ENSG00000142611", "ENSG00000253281", "ENSG00000125508"]  # Unseen test genes
        
    def verify_test_genes(self) -> bool:
        """Verify test genes are correctly categorized."""
        try:
            # Load training genes
            train_manifest = pd.read_csv(f'{self.training_dataset}/master/gene_manifest.csv')
            training_genes = set(train_manifest['gene_id'].tolist())
            
            # Verify Scenario 1 genes are in training
            for gene in self.scenario1_genes:
                if gene not in training_genes:
                    print(f"❌ ERROR: Scenario 1 gene {gene} not found in training data")
                    return False
            
            # Verify Scenario 2B genes are NOT in training
            for gene in self.scenario2b_genes:
                if gene in training_genes:
                    print(f"❌ ERROR: Scenario 2B gene {gene} found in training data")
                    return False
                    
            print("✅ Test gene categorization verified")
            return True
            
        except Exception as e:
            print(f"❌ Error verifying test genes: {e}")
            return False
    
    def run_test(self, test_name: str, genes: List[str], mode: str, 
                 output_dir: str, extra_args: List[str] = None) -> Dict[str, Any]:
        """Run a single inference test."""
        
        cmd = self.base_cmd + [
            "--model", self.model_path,
            "--training-dataset", self.training_dataset,
            "--genes", ",".join(genes),
            "--output-dir", f"results/{output_dir}",
            "--inference-mode", mode,
            "--verbose"
        ]
        
        if extra_args:
            cmd.extend(extra_args)
        
        print(f"\n🧪 Running {test_name}")
        print(f"   Mode: {mode}")
        print(f"   Genes: {genes}")
        print(f"   Output: results/{output_dir}")
        
        if self.dry_run:
            print(f"   [DRY RUN] Command: {' '.join(cmd)}")
            return {"status": "dry_run", "command": cmd}
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            end_time = time.time()
            
            duration = end_time - start_time
            success = result.returncode == 0
            
            test_result = {
                "status": "success" if success else "failed",
                "duration": duration,
                "returncode": result.returncode,
                "stdout_lines": len(result.stdout.split('\n')),
                "stderr_lines": len(result.stderr.split('\n'))
            }
            
            if success:
                print(f"   ✅ SUCCESS ({duration:.1f}s)")
                # Try to parse performance metrics
                try:
                    perf_file = Path(f"results/{output_dir}/performance_report.txt")
                    if perf_file.exists():
                        with open(perf_file) as f:
                            content = f.read()
                            test_result["performance_report"] = content
                except:
                    pass
            else:
                print(f"   ❌ FAILED ({duration:.1f}s)")
                print(f"   Return code: {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                    
                test_result["stderr"] = result.stderr
                
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT (>600s)")
            return {"status": "timeout"}
        except Exception as e:
            print(f"   💥 EXCEPTION: {e}")
            return {"status": "exception", "error": str(e)}
    
    def run_scenario1_tests(self) -> Dict[str, Any]:
        """Test Scenario 1: Genes with unseen positions (in training data)."""
        print("\n" + "="*60)
        print("🎯 SCENARIO 1: Genes with Unseen Positions")
        print("="*60)
        
        genes = self.scenario1_genes if not self.quick else self.scenario1_genes[:1]
        
        tests = {
            "scenario1_hybrid": self.run_test(
                "Scenario 1 - Hybrid Mode (Default)",
                genes, "hybrid", "test_scenario1_hybrid"
            ),
            "scenario1_base_only": self.run_test(
                "Scenario 1 - Base Only Mode", 
                genes, "base_only", "test_scenario1_base_only"
            ),
            "scenario1_meta_only": self.run_test(
                "Scenario 1 - Meta Only Mode",
                genes, "meta_only", "test_scenario1_meta_only"
            )
        }
        
        return tests
    
    def run_scenario2b_tests(self) -> Dict[str, Any]:
        """Test Scenario 2B: Completely unprocessed genes."""
        print("\n" + "="*60)
        print("🎯 SCENARIO 2B: Completely Unprocessed Genes")
        print("="*60)
        
        genes = self.scenario2b_genes if not self.quick else self.scenario2b_genes[:1]
        
        tests = {
            "scenario2b_hybrid": self.run_test(
                "Scenario 2B - Hybrid Mode (Default)",
                genes, "hybrid", "test_scenario2b_hybrid"
            ),
            "scenario2b_base_only": self.run_test(
                "Scenario 2B - Base Only Mode",
                genes, "base_only", "test_scenario2b_base_only"
            ),
            "scenario2b_meta_only": self.run_test(
                "Scenario 2B - Meta Only Mode",
                genes, "meta_only", "test_scenario2b_meta_only"
            )
        }
        
        return tests
    
    def run_mixed_tests(self) -> Dict[str, Any]:
        """Test mixed scenarios to validate scenario detection."""
        print("\n" + "="*60)
        print("🎯 MIXED SCENARIOS: Scenario Detection")
        print("="*60)
        
        # Mix of Scenario 1 and 2B genes
        mixed_genes = [
            self.scenario1_genes[0],  # In training
            self.scenario2b_genes[0]  # Not in training
        ]
        
        tests = {
            "mixed_hybrid": self.run_test(
                "Mixed Scenarios - Hybrid Mode",
                mixed_genes, "hybrid", "test_mixed_hybrid"
            )
        }
        
        return tests
    
    def validate_results(self) -> Dict[str, Any]:
        """Validate test results against expected behavior."""
        print("\n" + "="*60)
        print("📊 VALIDATING RESULTS")
        print("="*60)
        
        validation = {
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "mode_validation": {},
            "scenario_validation": {},
            "integration_validation": {}
        }
        
        for category, tests in self.results.items():
            for test_name, result in tests.items():
                validation["total_tests"] += 1
                if result.get("status") == "success":
                    validation["successful_tests"] += 1
                else:
                    validation["failed_tests"] += 1
        
        # Validate inference mode behavior
        print("\n🎯 Mode Validation:")
        for mode in ["hybrid", "base_only", "meta_only"]:
            mode_tests = [t for tests in self.results.values() for t in tests.keys() if mode in t]
            mode_successes = sum(1 for tests in self.results.values() 
                               for name, result in tests.items() 
                               if mode in name and result.get("status") == "success")
            
            validation["mode_validation"][mode] = {
                "tests": len(mode_tests),
                "successes": mode_successes,
                "success_rate": mode_successes / len(mode_tests) if mode_tests else 0
            }
            
            print(f"  {mode}: {mode_successes}/{len(mode_tests)} tests passed")
        
        # Validate scenario behavior
        print("\n🧬 Scenario Validation:")
        for scenario in ["scenario1", "scenario2b", "mixed"]:
            scenario_tests = [t for tests in self.results.values() for t in tests.keys() if scenario in t]
            scenario_successes = sum(1 for tests in self.results.values() 
                                   for name, result in tests.items() 
                                   if scenario in name and result.get("status") == "success")
            
            validation["scenario_validation"][scenario] = {
                "tests": len(scenario_tests),
                "successes": scenario_successes,
                "success_rate": scenario_successes / len(scenario_tests) if scenario_tests else 0
            }
            
            print(f"  {scenario}: {scenario_successes}/{len(scenario_tests)} tests passed")
        
        return validation
    
    def generate_report(self, validation: Dict[str, Any]) -> None:
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        # Summary
        total = validation["total_tests"]
        success = validation["successful_tests"]
        failed = validation["failed_tests"]
        success_rate = (success / total * 100) if total > 0 else 0
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"  Total tests: {total}")
        print(f"  Successful: {success} ({success_rate:.1f}%)")
        print(f"  Failed: {failed}")
        print(f"  Status: {'✅ PASS' if failed == 0 else '❌ FAIL'}")
        
        # Mode-specific results
        print(f"\n🎯 INFERENCE MODE VALIDATION:")
        for mode, stats in validation["mode_validation"].items():
            rate = stats["success_rate"] * 100
            status = "✅" if stats["successes"] == stats["tests"] else "❌"
            print(f"  {mode}: {stats['successes']}/{stats['tests']} ({rate:.1f}%) {status}")
        
        # Expected behavior validation
        print(f"\n🔍 EXPECTED BEHAVIOR VALIDATION:")
        
        # Check if hybrid is default
        hybrid_tests = validation["mode_validation"].get("hybrid", {}).get("tests", 0)
        print(f"  Hybrid mode tested: {'✅' if hybrid_tests > 0 else '❌'}")
        
        # Check scenario coverage
        s1_tests = validation["scenario_validation"].get("scenario1", {}).get("tests", 0)
        s2b_tests = validation["scenario_validation"].get("scenario2b", {}).get("tests", 0)
        print(f"  Scenario 1 coverage: {'✅' if s1_tests >= 3 else '❌'} ({s1_tests} tests)")
        print(f"  Scenario 2B coverage: {'✅' if s2b_tests >= 3 else '❌'} ({s2b_tests} tests)")
        
        # Save detailed results
        report_file = Path("results/inference_mode_test_results.json")
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump({
                "results": self.results,
                "validation": validation,
                "timestamp": time.time()
            }, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: {report_file}")
    
    def run_all_tests(self) -> None:
        """Run the complete test suite."""
        print("🧪 COMPREHENSIVE INFERENCE MODE TEST SUITE")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Training Dataset: {self.training_dataset}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE EXECUTION'}")
        print(f"Speed: {'QUICK' if self.quick else 'FULL'}")
        
        if not self.verify_test_genes():
            print("❌ Test gene verification failed. Aborting.")
            return
        
        # Run all test categories
        self.results["scenario1"] = self.run_scenario1_tests()
        self.results["scenario2b"] = self.run_scenario2b_tests()
        self.results["mixed"] = self.run_mixed_tests()
        
        # Validate and report
        validation = self.validate_results()
        self.generate_report(validation)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test all inference modes comprehensively")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show commands without executing them")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with fewer genes")
    
    args = parser.parse_args()
    
    test_suite = InferenceModeTestSuite(dry_run=args.dry_run, quick=args.quick)
    test_suite.run_all_tests()

if __name__ == "__main__":
    main()