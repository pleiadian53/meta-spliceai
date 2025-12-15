#!/usr/bin/env python3
"""
Universal Training Monitor for Meta-SpliceAI's Meta-Model Training

This script monitors both single-instance and multi-instance ensemble training,
providing comprehensive progress tracking, error detection, and output validation.

Usage:
    python monitor_training_universal.py --run-name <run_name>
    python monitor_training_universal.py --auto-detect
    python monitor_training_universal.py --list-runs
"""

import argparse
import subprocess
import time
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class TrainingMonitor:
    """Universal monitor for splice surveyor training runs."""
    
    def __init__(self, run_name: str, verbose: bool = True):
        self.run_name = run_name
        self.verbose = verbose
        self.log_file = Path(f"logs/{run_name}.log")
        self.results_dir = Path(f"results/{run_name}")
        
    def detect_training_mode(self) -> str:
        """Detect if this is single-instance or multi-instance training."""
        if not self.log_file.exists():
            return "unknown"
        
        try:
            with open(self.log_file, 'r') as f:
                content = f.read()
            
            if "Multi-Instance Ensemble" in content:
                return "multi_instance"
            elif "Single Model Training" in content:
                return "single_instance"
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    def check_process_status(self) -> Dict[str, any]:
        """Check if training process is running and get resource usage."""
        try:
            # Look for the training process
            result = subprocess.run(
                ["pgrep", "-f", f"run_gene_cv_sigmoid.*{self.run_name}"], 
                capture_output=True, text=True, timeout=5
            )
            
            if result.stdout.strip():
                pid = result.stdout.strip().split('\n')[0]
                
                # Get detailed process info
                ps_result = subprocess.run(
                    ["ps", "-p", pid, "-o", "pid,etime,rss,%cpu,%mem,stat"],
                    capture_output=True, text=True, timeout=5
                )
                
                if ps_result.returncode == 0:
                    lines = ps_result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        fields = lines[1].split()
                        return {
                            'running': True,
                            'pid': pid,
                            'runtime': fields[1] if len(fields) > 1 else 'unknown',
                            'memory_mb': int(fields[2]) / 1024 if len(fields) > 2 else 0,
                            'cpu_percent': fields[3] if len(fields) > 3 else 'unknown',
                            'mem_percent': fields[4] if len(fields) > 4 else 'unknown',
                            'status': fields[5] if len(fields) > 5 else 'unknown'
                        }
            
            return {'running': False}
            
        except Exception as e:
            return {'running': False, 'error': str(e)}
    
    def analyze_log_progress(self) -> Dict[str, any]:
        """Analyze log file for progress indicators and issues."""
        if not self.log_file.exists():
            return {'error': 'Log file not found'}
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Training mode detection
            training_mode = self.detect_training_mode()
            
            # Define milestones based on training mode
            if training_mode == "multi_instance":
                milestones = [
                    (r"🔥.*Multi-Instance.*Training.*(\d+).*instances", "Multi-instance started"),
                    (r"🔧.*Instance.*(\d+)/(\d+).*Training", "Instance training"),
                    (r"✅.*Instance.*(\d+).*completed", "Instance completed"),
                    (r"🔄.*Consolidating.*(\d+).*instances", "Consolidation started"),
                    (r"✅.*Consolidated model created", "Consolidation completed"),
                    (r"📊.*post-training analysis", "Post-training started"),
                    (r"🧠.*SHAP.*analysis", "SHAP analysis"),
                    (r"✓.*Comprehensive.*SHAP", "Enhanced SHAP success"),
                    (r"🎉.*Training.*completed", "Training completed")
                ]
            else:
                milestones = [
                    (r"🚀.*Single Model Training", "Single model started"),
                    (r"🔀.*Running.*cross-validation.*(\d+).*folds", "Cross-validation started"),
                    (r"Fold.*(\d+)/(\d+)", "CV fold progress"),
                    (r"📊.*Cross-Validation Summary", "CV completed"),
                    (r"🎯.*Training final model", "Final model training"),
                    (r"💾.*model.*saved", "Model saved"),
                    (r"📊.*post-training analysis", "Post-training started"),
                    (r"🧠.*SHAP.*analysis", "SHAP analysis"),
                    (r"🎉.*completed", "Training completed")
                ]
            
            # Count milestone occurrences
            milestone_counts = {}
            for pattern, desc in milestones:
                matches = [line for line in lines if re.search(pattern, line, re.IGNORECASE)]
                if matches:
                    milestone_counts[desc] = len(matches)
            
            # Look for errors and warnings
            error_patterns = [
                (r"❌|✗.*failed|Exception|Error|Traceback", "Errors"),
                (r"⚠️|Warning|fallback|falling back", "Warnings"),
                (r"killed|terminated|segmentation|out of memory", "Critical issues")
            ]
            
            issue_counts = {}
            recent_issues = []
            
            for pattern, desc in error_patterns:
                matches = [(i, line) for i, line in enumerate(lines) if re.search(pattern, line, re.IGNORECASE)]
                if matches:
                    issue_counts[desc] = len(matches)
                    # Keep last 3 issues
                    recent_issues.extend([(desc, line.strip()) for _, line in matches[-3:]])
            
            # Recent activity (last 10 lines, filtered)
            recent_activity = []
            for line in lines[-10:]:
                clean_line = line.strip()
                if (clean_line and 
                    not clean_line.startswith('E0000') and 
                    not clean_line.startswith('W0000') and
                    not clean_line.startswith('WARNING: All log') and
                    len(clean_line) > 10):
                    recent_activity.append(clean_line)
            
            return {
                'training_mode': training_mode,
                'total_lines': len(lines),
                'milestones': milestone_counts,
                'issues': issue_counts,
                'recent_issues': recent_issues[-5:],  # Last 5 issues
                'recent_activity': recent_activity[-5:],  # Last 5 meaningful lines
                'last_update': datetime.fromtimestamp(self.log_file.stat().st_mtime)
            }
            
        except Exception as e:
            return {'error': f'Failed to analyze log: {e}'}
    
    def analyze_outputs(self) -> Dict[str, any]:
        """Analyze generated output files and directories."""
        if not self.results_dir.exists():
            return {'exists': False}
        
        # Count file types
        file_counts = {
            'pkl_files': len(list(self.results_dir.glob("**/*.pkl"))),
            'json_files': len(list(self.results_dir.glob("**/*.json"))),
            'csv_files': len(list(self.results_dir.glob("**/*.csv"))),
            'pdf_files': len(list(self.results_dir.glob("**/*.pdf"))),
            'png_files': len(list(self.results_dir.glob("**/*.png"))),
        }
        
        # Check for key directories
        key_directories = [
            "multi_instance_training",
            "cv_metrics_visualization", 
            "feature_importance_analysis",
            "leakage_analysis",
            "comprehensive_shap_analysis"
        ]
        
        directory_status = {}
        for dir_name in key_directories:
            dir_path = self.results_dir / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.glob("**/*")))
                directory_status[dir_name] = {'exists': True, 'files': file_count}
            else:
                directory_status[dir_name] = {'exists': False, 'files': 0}
        
        # Check for critical files
        critical_files = {
            'main_model': self.results_dir / "model_multiclass.pkl",
            'consolidation_info': self.results_dir / "consolidation_info.json",
            'training_results': self.results_dir / "complete_training_results.json",
            'cv_metrics': self.results_dir / "gene_cv_metrics.csv",
            'feature_manifest': self.results_dir / "feature_manifest.csv"
        }
        
        file_status = {}
        for file_key, file_path in critical_files.items():
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                file_status[file_key] = {'exists': True, 'size_mb': size_mb}
            else:
                file_status[file_key] = {'exists': False, 'size_mb': 0}
        
        # Multi-instance specific analysis
        multi_instance_info = {}
        if directory_status['multi_instance_training']['exists']:
            instance_dirs = list((self.results_dir / "multi_instance_training").glob("instance_*"))
            instance_models = []
            
            for instance_dir in sorted(instance_dirs):
                model_file = instance_dir / "model_multiclass.pkl"
                if model_file.exists():
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    instance_models.append({
                        'instance': instance_dir.name,
                        'model_size_mb': size_mb,
                        'has_metrics': (instance_dir / "metrics_aggregate.json").exists(),
                        'has_cv_results': (instance_dir / "gene_cv_metrics.csv").exists()
                    })
            
            multi_instance_info = {
                'total_instances': len(instance_dirs),
                'completed_instances': len(instance_models),
                'instance_details': instance_models
            }
        
        return {
            'exists': True,
            'file_counts': file_counts,
            'directories': directory_status,
            'critical_files': file_status,
            'multi_instance': multi_instance_info
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive training status report."""
        
        # Get all analysis data
        process_status = self.check_process_status()
        log_analysis = self.analyze_log_progress()
        output_analysis = self.analyze_outputs()
        
        # Generate report
        report_lines = []
        report_lines.append(f"🔍 Training Monitor Report: {self.run_name}")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Process status
        report_lines.append("🔄 Process Status:")
        if process_status.get('running'):
            report_lines.append(f"  ✅ Running (PID: {process_status['pid']})")
            report_lines.append(f"  ⏰ Runtime: {process_status.get('runtime', 'unknown')}")
            report_lines.append(f"  💾 Memory: {process_status.get('memory_mb', 0):.1f} MB")
            report_lines.append(f"  🔧 CPU: {process_status.get('cpu_percent', 'unknown')}%")
        else:
            report_lines.append("  ❌ Not running")
            if 'error' in process_status:
                report_lines.append(f"  Error: {process_status['error']}")
        
        report_lines.append("")
        
        # Log analysis
        if 'error' not in log_analysis:
            training_mode = log_analysis.get('training_mode', 'unknown')
            report_lines.append(f"📊 Training Analysis (Mode: {training_mode.title()}):")
            report_lines.append(f"  📝 Log lines: {log_analysis.get('total_lines', 0)}")
            
            if log_analysis.get('last_update'):
                time_since_update = datetime.now() - log_analysis['last_update']
                report_lines.append(f"  🕐 Last update: {time_since_update.total_seconds() / 60:.1f} minutes ago")
            
            # Milestones
            milestones = log_analysis.get('milestones', {})
            if milestones:
                report_lines.append("  ✅ Completed milestones:")
                for milestone, count in milestones.items():
                    report_lines.append(f"    - {milestone}: {count}")
            
            # Issues
            issues = log_analysis.get('issues', {})
            if issues:
                report_lines.append("  ⚠️  Issues detected:")
                for issue_type, count in issues.items():
                    report_lines.append(f"    - {issue_type}: {count}")
            
            # Recent activity
            recent_activity = log_analysis.get('recent_activity', [])
            if recent_activity:
                report_lines.append("  📝 Recent activity:")
                for activity in recent_activity:
                    report_lines.append(f"    {activity[:100]}...")
            
        else:
            report_lines.append(f"📊 Log Analysis: {log_analysis['error']}")
        
        report_lines.append("")
        
        # Output analysis
        if output_analysis.get('exists'):
            report_lines.append("📁 Output Analysis:")
            
            # File counts
            file_counts = output_analysis['file_counts']
            report_lines.append(f"  📦 Files: {file_counts['pkl_files']} models, {file_counts['json_files']} metadata, "
                              f"{file_counts['csv_files']} data, {file_counts['pdf_files']} plots")
            
            # Critical files
            critical_files = output_analysis['critical_files']
            report_lines.append("  🎯 Critical files:")
            for file_key, file_info in critical_files.items():
                status = "✅" if file_info['exists'] else "❌"
                size_info = f" ({file_info['size_mb']:.1f}MB)" if file_info['exists'] else ""
                report_lines.append(f"    {status} {file_key}{size_info}")
            
            # Directories
            directories = output_analysis['directories']
            report_lines.append("  📂 Key directories:")
            for dir_name, dir_info in directories.items():
                status = "✅" if dir_info['exists'] else "❌"
                file_info = f" ({dir_info['files']} files)" if dir_info['exists'] else ""
                report_lines.append(f"    {status} {dir_name}{file_info}")
            
            # Multi-instance specific info
            multi_info = output_analysis.get('multi_instance', {})
            if multi_info:
                report_lines.append("  🔢 Multi-instance details:")
                report_lines.append(f"    Instances: {multi_info['completed_instances']}/{multi_info['total_instances']} completed")
                
                if multi_info['instance_details']:
                    report_lines.append("    Instance status:")
                    for instance in multi_info['instance_details'][-3:]:  # Show last 3
                        report_lines.append(f"      {instance['instance']}: {instance['model_size_mb']:.1f}MB")
        else:
            report_lines.append("📁 Output Analysis: Results directory not created yet")
        
        return "\n".join(report_lines)
    
    def print_report(self):
        """Print the comprehensive training report."""
        print(self.generate_report())


def auto_detect_active_runs() -> List[str]:
    """Auto-detect active training runs."""
    try:
        # Look for active processes
        result = subprocess.run(
            ["pgrep", "-f", "run_gene_cv_sigmoid"], 
            capture_output=True, text=True, timeout=5
        )
        
        if not result.stdout.strip():
            return []
        
        # Get command lines for active processes
        active_runs = []
        for pid in result.stdout.strip().split('\n'):
            try:
                cmd_result = subprocess.run(
                    ["ps", "-p", pid, "-o", "args="],
                    capture_output=True, text=True, timeout=5
                )
                
                if cmd_result.returncode == 0:
                    cmd_line = cmd_result.stdout.strip()
                    # Extract run name from --out-dir argument
                    match = re.search(r'--out-dir\s+results/([^\s]+)', cmd_line)
                    if match:
                        run_name = match.group(1)
                        active_runs.append(run_name)
            except:
                continue
        
        return active_runs
        
    except Exception:
        return []


def list_all_runs() -> List[Tuple[str, str, bool]]:
    """List all available training runs."""
    results_dir = Path("results")
    logs_dir = Path("logs")
    
    runs = []
    
    # Find all result directories
    if results_dir.exists():
        for run_dir in results_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith(('gene_cv', 'test_')):
                log_file = logs_dir / f"{run_dir.name}.log"
                has_log = log_file.exists()
                
                # Determine training mode
                mode = "unknown"
                if has_log:
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read(2000)  # Read first 2000 chars for better detection
                        if "Multi-Instance Ensemble" in content:
                            mode = "multi_instance"
                        elif "Single Model Training" in content or "Gene-CV-Sigmoid" in content:
                            mode = "single_instance"
                        elif "Batch Ensemble" in content:
                            mode = "batch_ensemble"
                    except:
                        pass
                
                runs.append((run_dir.name, mode, has_log))
    
    return sorted(runs)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Universal Training Monitor for MetaSpliceAI")
    parser.add_argument("--run-name", help="Specific run name to monitor")
    parser.add_argument("--auto-detect", action="store_true", help="Auto-detect and monitor active runs")
    parser.add_argument("--list-runs", action="store_true", help="List all available runs")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring (updates every 30 seconds)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.list_runs:
        print("📋 Available Training Runs:")
        print("=" * 60)
        
        runs = list_all_runs()
        if not runs:
            print("No training runs found")
            return
        
        for run_name, mode, has_log in runs:
            status = "📝" if has_log else "❌"
            print(f"  {status} {run_name:<40} ({mode})")
        
        print(f"\nTotal: {len(runs)} runs found")
        return
    
    if args.auto_detect:
        active_runs = auto_detect_active_runs()
        
        if not active_runs:
            print("❌ No active training processes found")
            print("💡 Use --list-runs to see all available runs")
            return
        
        print(f"🔍 Found {len(active_runs)} active training run(s):")
        for run_name in active_runs:
            print(f"  📊 Monitoring: {run_name}")
            monitor = TrainingMonitor(run_name, verbose=args.verbose)
            monitor.print_report()
            print()
        
        return
    
    if not args.run_name:
        parser.error("Must specify --run-name, --auto-detect, or --list-runs")
    
    # Monitor specific run
    monitor = TrainingMonitor(args.run_name, verbose=args.verbose)
    
    if args.watch:
        print(f"🔍 Watching training run: {args.run_name}")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 60)
        
        try:
            while True:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
                monitor.print_report()
                print("\n" + "─" * 60)
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
    else:
        monitor.print_report()


if __name__ == "__main__":
    main()
