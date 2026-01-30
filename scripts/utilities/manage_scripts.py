#!/usr/bin/env python3
"""
MetaSpliceAI Script Management Utility

This utility helps manage the growing collection of scripts by providing:
1. Script discovery and cataloging
2. Usage tracking and recommendations
3. Documentation validation
4. Maintenance helpers

Usage:
    python scripts/manage_scripts.py --list-all
    python scripts/manage_scripts.py --find "gpu"
    python scripts/manage_scripts.py --validate-docs
    python scripts/manage_scripts.py --suggest-for "cv analysis"
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re
import json
from datetime import datetime

# Script categories and their descriptions
SCRIPT_CATEGORIES = {
    'setup': {
        'description': 'Environment setup and system validation',
        'keywords': ['gpu', 'install', 'setup', 'check', 'diagnose', 'environment'],
        'priority': 'low'  # Used infrequently
    },
    'data': {
        'description': 'Data preparation, validation, and analysis',
        'keywords': ['data', 'validate', 'analyze', 'inspect', 'sequence', 'training'],
        'priority': 'high'  # Used frequently
    },
    'evaluation': {
        'description': 'Model evaluation and performance analysis',
        'keywords': ['evaluation', 'cv', 'f1', 'pr', 'metrics', 'performance'],
        'priority': 'high'  # Used frequently
    },
    'training': {
        'description': 'Model training and scaling',
        'keywords': ['training', 'gpu', 'multi', 'scaling', 'model'],
        'priority': 'medium'  # Used per project
    },
    'utility': {
        'description': 'General utilities and maintenance',
        'keywords': ['cleanup', 'validate', 'artifact', 'patch', 'fix'],
        'priority': 'medium'  # Used as needed
    },
    'visualization': {
        'description': 'Plotting and visualization',
        'keywords': ['plot', 'diagram', 'visualization', 'chart'],
        'priority': 'medium'  # Used per project
    }
}

# High-priority scripts that are used frequently
HIGH_PRIORITY_SCRIPTS = [
    'f1_pr_analysis_merged.py',
    'pre_flight_checks.py', 
    'analyze_training_data.py',
    'validate_meta_model_training_data.py'
]

class ScriptManager:
    def __init__(self, scripts_dir: Path):
        self.scripts_dir = scripts_dir
        self.scripts = self._discover_scripts()
    
    def _discover_scripts(self) -> List[Dict]:
        """Discover all Python scripts in the scripts directory."""
        scripts = []
        
        for root, dirs, files in os.walk(self.scripts_dir):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    script_path = Path(root) / file
                    relative_path = script_path.relative_to(self.scripts_dir)
                    
                    script_info = {
                        'name': file,
                        'path': str(relative_path),
                        'full_path': script_path,
                        'category': self._categorize_script(file, str(relative_path)),
                        'priority': self._get_priority(file),
                        'description': self._extract_description(script_path),
                        'size_kb': script_path.stat().st_size / 1024,
                        'modified': datetime.fromtimestamp(script_path.stat().st_mtime)
                    }
                    scripts.append(script_info)
        
        return sorted(scripts, key=lambda x: (x['priority'], x['name']))
    
    def _categorize_script(self, filename: str, path: str) -> str:
        """Categorize a script based on its name and path."""
        filename_lower = filename.lower()
        path_lower = path.lower()
        
        # Check path-based categories first
        if 'evaluation' in path_lower:
            return 'evaluation'
        if 'analysis' in path_lower:
            return 'data'
        if 'gpu' in path_lower or 'installation' in path_lower:
            return 'setup'
        if 'scaling' in path_lower:
            return 'training'
        
        # Check filename-based categories
        for category, info in SCRIPT_CATEGORIES.items():
            for keyword in info['keywords']:
                if keyword in filename_lower:
                    return category
        
        return 'utility'  # Default category
    
    def _get_priority(self, filename: str) -> int:
        """Get priority score for sorting (0=highest priority)."""
        if filename in HIGH_PRIORITY_SCRIPTS:
            return 0  # Highest priority
        
        category = self._categorize_script(filename, '')
        category_info = SCRIPT_CATEGORIES.get(category, {})
        priority = category_info.get('priority', 'medium')
        
        priority_scores = {'high': 1, 'medium': 2, 'low': 3}
        return priority_scores.get(priority, 2)
    
    def _extract_description(self, script_path: Path) -> str:
        """Extract description from script docstring."""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for docstring after shebang
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1).strip()
                # Get first meaningful line
                lines = [line.strip() for line in docstring.split('\n') if line.strip()]
                if lines:
                    return lines[0][:100] + ('...' if len(lines[0]) > 100 else '')
            
            return "No description available"
        except Exception:
            return "Description unavailable"
    
    def list_all_scripts(self) -> None:
        """List all scripts organized by category."""
        print("🗂️  MetaSpliceAI Script Catalog")
        print("=" * 50)
        
        by_category = {}
        for script in self.scripts:
            category = script['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(script)
        
        for category, scripts in by_category.items():
            category_info = SCRIPT_CATEGORIES.get(category, {})
            priority_icon = {'high': '🌟', 'medium': '📋', 'low': '🔧'}.get(
                category_info.get('priority', 'medium'), '📋'
            )
            
            print(f"\n{priority_icon} {category.upper()}")
            print(f"   {category_info.get('description', 'No description')}")
            print("-" * 40)
            
            for script in scripts:
                priority_marker = "⭐" if script['priority'] == 0 else "  "
                print(f"{priority_marker} {script['name']:<30} {script['path']}")
                print(f"     {script['description']}")
                print()
    
    def find_scripts(self, query: str) -> List[Dict]:
        """Find scripts matching a query."""
        query_lower = query.lower()
        matches = []
        
        for script in self.scripts:
            # Search in name, path, and description
            if (query_lower in script['name'].lower() or 
                query_lower in script['path'].lower() or 
                query_lower in script['description'].lower()):
                matches.append(script)
        
        return matches
    
    def suggest_scripts(self, task_description: str) -> List[Dict]:
        """Suggest scripts for a given task."""
        task_lower = task_description.lower()
        suggestions = []
        
        # High-priority matches
        for script in self.scripts:
            if script['priority'] == 0:  # High-priority scripts
                for keyword in task_lower.split():
                    if (keyword in script['name'].lower() or 
                        keyword in script['description'].lower()):
                        suggestions.append((script, 'high'))
                        break
        
        # Category-based matches
        for category, info in SCRIPT_CATEGORIES.items():
            for keyword in info['keywords']:
                if keyword in task_lower:
                    category_scripts = [s for s in self.scripts if s['category'] == category]
                    for script in category_scripts[:3]:  # Top 3 from category
                        if script not in [s[0] for s in suggestions]:
                            suggestions.append((script, 'medium'))
        
        return suggestions
    
    def validate_documentation(self) -> Dict[str, List[str]]:
        """Validate that scripts have proper documentation."""
        issues = {
            'missing_docstring': [],
            'missing_usage': [],
            'large_undocumented': [],
            'high_priority_undocumented': []
        }
        
        for script in self.scripts:
            # Check for docstring
            if script['description'] == "No description available":
                issues['missing_docstring'].append(script['name'])
            
            # Check for usage examples in high-priority scripts
            if script['priority'] == 0:
                try:
                    with open(script['full_path'], 'r') as f:
                        content = f.read()
                    if 'usage:' not in content.lower() and 'example:' not in content.lower():
                        issues['missing_usage'].append(script['name'])
                except Exception:
                    pass
            
            # Check large scripts without documentation
            if script['size_kb'] > 10 and script['description'] == "No description available":
                issues['large_undocumented'].append(f"{script['name']} ({script['size_kb']:.1f}KB)")
        
        return issues
    
    def generate_usage_stats(self) -> Dict:
        """Generate usage statistics."""
        stats = {
            'total_scripts': len(self.scripts),
            'by_category': {},
            'by_priority': {},
            'largest_scripts': [],
            'recently_modified': []
        }
        
        # Category stats
        for script in self.scripts:
            category = script['category']
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
        
        # Priority stats
        priority_names = {0: 'high', 1: 'medium', 2: 'low', 3: 'very_low'}
        for script in self.scripts:
            priority_name = priority_names.get(script['priority'], 'unknown')
            stats['by_priority'][priority_name] = stats['by_priority'].get(priority_name, 0) + 1
        
        # Largest scripts
        stats['largest_scripts'] = sorted(
            [(s['name'], s['size_kb']) for s in self.scripts],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        # Recently modified
        stats['recently_modified'] = sorted(
            [(s['name'], s['modified']) for s in self.scripts],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='MetaSpliceAI Script Management Utility')
    parser.add_argument('--list-all', action='store_true', help='List all scripts by category')
    parser.add_argument('--find', type=str, help='Find scripts matching query')
    parser.add_argument('--suggest-for', type=str, help='Suggest scripts for a task')
    parser.add_argument('--validate-docs', action='store_true', help='Validate script documentation')
    parser.add_argument('--stats', action='store_true', help='Show usage statistics')
    
    args = parser.parse_args()
    
    scripts_dir = Path(__file__).parent
    manager = ScriptManager(scripts_dir)
    
    if args.list_all:
        manager.list_all_scripts()
    
    elif args.find:
        matches = manager.find_scripts(args.find)
        print(f"🔍 Found {len(matches)} scripts matching '{args.find}':")
        print("-" * 40)
        for script in matches:
            print(f"⭐ {script['name']} ({script['path']})")
            print(f"   {script['description']}")
            print()
    
    elif args.suggest_for:
        suggestions = manager.suggest_scripts(args.suggest_for)
        print(f"💡 Script suggestions for '{args.suggest_for}':")
        print("-" * 40)
        
        if not suggestions:
            print("No specific suggestions found. Try:")
            print("- python scripts/manage_scripts.py --find <keyword>")
            print("- python scripts/manage_scripts.py --list-all")
        else:
            for script, confidence in suggestions:
                confidence_icon = "🌟" if confidence == 'high' else "📋"
                print(f"{confidence_icon} {script['name']}")
                print(f"   Command: python scripts/{script['path']}")
                print(f"   {script['description']}")
                print()
    
    elif args.validate_docs:
        issues = manager.validate_documentation()
        print("📋 Documentation Validation Report")
        print("=" * 40)
        
        for issue_type, scripts in issues.items():
            if scripts:
                print(f"\n❌ {issue_type.replace('_', ' ').title()}:")
                for script in scripts:
                    print(f"   - {script}")
        
        if not any(issues.values()):
            print("✅ All scripts have adequate documentation!")
    
    elif args.stats:
        stats = manager.generate_usage_stats()
        print("📊 Script Usage Statistics")
        print("=" * 30)
        print(f"Total Scripts: {stats['total_scripts']}")
        
        print("\nBy Category:")
        for category, count in stats['by_category'].items():
            print(f"  {category}: {count}")
        
        print("\nBy Priority:")
        for priority, count in stats['by_priority'].items():
            print(f"  {priority}: {count}")
        
        print("\nLargest Scripts:")
        for name, size in stats['largest_scripts']:
            print(f"  {name}: {size:.1f}KB")
    
    else:
        print("MetaSpliceAI Script Manager")
        print("Use --help for available options")
        print("\nQuick commands:")
        print("  --list-all              # See all scripts")
        print("  --find 'gpu'           # Find GPU-related scripts")
        print("  --suggest-for 'cv analysis'  # Get suggestions")
        print("  --validate-docs        # Check documentation")

if __name__ == "__main__":
    main()
