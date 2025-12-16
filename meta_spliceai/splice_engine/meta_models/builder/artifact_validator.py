"""
Artifact Schema Validator for Meta-Model Training Data
=====================================================

This module provides utilities to validate the schema and integrity of artifacts
generated during the meta-model training data assembly process. It can identify
corrupted files, files with wrong schemas, and provide detailed reports on
artifact health.

The validator checks three main artifact types:
1. analysis_sequences_*_chunk_*.tsv - Detailed sequence analysis data
2. splice_positions_enhanced_*_chunk_*.tsv - Enhanced splice position data  
3. splice_errors_*_chunk_*.tsv - Error information

Usage:
    python -m meta_spliceai.splice_engine.meta_models.builder.artifact_validator
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import argparse
import json

import polars as pl
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from meta_spliceai.system.config import Config


class ArtifactType(Enum):
    """Enumeration of artifact types."""
    ANALYSIS_SEQUENCES = "analysis_sequences"
    SPLICE_POSITIONS_ENHANCED = "splice_positions_enhanced"
    SPLICE_ERRORS = "splice_errors"


@dataclass
class SchemaDefinition:
    """Schema definition for an artifact type."""
    artifact_type: ArtifactType
    required_columns: Set[str]
    optional_columns: Set[str]
    data_types: Dict[str, str]
    min_rows: int = 1
    max_file_size_mb: Optional[float] = None


@dataclass
class ValidationResult:
    """Result of artifact validation."""
    file_path: Path
    artifact_type: ArtifactType
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    row_count: int
    column_count: int
    file_size_mb: float
    schema_matches: bool
    data_types_valid: bool
    has_required_columns: bool


class ArtifactValidator:
    """Validator for meta-model training artifacts."""
    
    def __init__(self, artifacts_dir: Optional[Path] = None):
        """
        Initialize the validator.
        
        Parameters
        ----------
        artifacts_dir
            Directory containing artifacts. If None, uses default from Config.
        """
        if artifacts_dir is None:
            try:
                artifacts_dir = Path(Config.DATA_DIR) / "ensembl" / "spliceai_eval" / "meta_models"
            except Exception:
                # Fallback to relative path
                artifacts_dir = Path("data/ensembl/spliceai_eval/meta_models")
        
        self.artifacts_dir = Path(artifacts_dir)
        self.schema_definitions = self._define_schemas()
    
    def _define_schemas(self) -> Dict[ArtifactType, SchemaDefinition]:
        """Define expected schemas for each artifact type."""
        return {
            ArtifactType.ANALYSIS_SEQUENCES: SchemaDefinition(
                artifact_type=ArtifactType.ANALYSIS_SEQUENCES,
                required_columns={
                    "gene_id", "transcript_id", "predicted_position", "position", 
                    "score", "strand", "neither_score", "true_position", 
                    "context_score_p2", "splice_type", "acceptor_score", 
                    "context_score_m2", "context_score_p1", "context_score_m1", 
                    "donor_score", "pred_type", "relative_donor_probability", 
                    "splice_probability", "donor_acceptor_diff", "splice_neither_diff", 
                    "donor_acceptor_logodds", "splice_neither_logodds", 
                    "probability_entropy", "context_neighbor_mean", 
                    "context_asymmetry", "context_max", "donor_diff_m1", 
                    "donor_diff_m2", "donor_diff_p1", "donor_diff_p2", 
                    "donor_surge_ratio", "donor_is_local_peak", 
                    "donor_weighted_context", "donor_peak_height_ratio", 
                    "donor_second_derivative", "donor_signal_strength", 
                    "donor_context_diff_ratio", "acceptor_diff_m1", 
                    "acceptor_diff_m2", "acceptor_diff_p1", "acceptor_diff_p2", 
                    "acceptor_surge_ratio", "acceptor_is_local_peak", 
                    "acceptor_weighted_context", "acceptor_peak_height_ratio", 
                    "acceptor_second_derivative", "acceptor_signal_strength", 
                    "acceptor_context_diff_ratio", "donor_acceptor_peak_ratio", 
                    "type_signal_difference", "score_difference_ratio", 
                    "signal_strength_ratio", "chrom", "window_start", 
                    "window_end", "transcript_count", "sequence"
                },
                optional_columns=set(),
                data_types={
                    "gene_id": "str",
                    "transcript_id": "str", 
                    "predicted_position": "int",
                    "position": "int",
                    "score": "float",
                    "strand": "str",
                    "neither_score": "float",
                    "true_position": "int",
                    "context_score_p2": "float",
                    "splice_type": "str",
                    "acceptor_score": "float",
                    "context_score_m2": "float",
                    "context_score_p1": "float",
                    "context_score_m1": "float",
                    "donor_score": "float",
                    "pred_type": "str",
                    "relative_donor_probability": "float",
                    "splice_probability": "float",
                    "donor_acceptor_diff": "float",
                    "splice_neither_diff": "float",
                    "donor_acceptor_logodds": "float",
                    "splice_neither_logodds": "float",
                    "probability_entropy": "float",
                    "context_neighbor_mean": "float",
                    "context_asymmetry": "float",
                    "context_max": "float",
                    "donor_diff_m1": "float",
                    "donor_diff_m2": "float",
                    "donor_diff_p1": "float",
                    "donor_diff_p2": "float",
                    "donor_surge_ratio": "float",
                    "donor_is_local_peak": "int",
                    "donor_weighted_context": "float",
                    "donor_peak_height_ratio": "float",
                    "donor_second_derivative": "float",
                    "donor_signal_strength": "float",
                    "donor_context_diff_ratio": "float",
                    "acceptor_diff_m1": "float",
                    "acceptor_diff_m2": "float",
                    "acceptor_diff_p1": "float",
                    "acceptor_diff_p2": "float",
                    "acceptor_surge_ratio": "float",
                    "acceptor_is_local_peak": "int",
                    "acceptor_weighted_context": "float",
                    "acceptor_peak_height_ratio": "float",
                    "acceptor_second_derivative": "float",
                    "acceptor_signal_strength": "float",
                    "acceptor_context_diff_ratio": "float",
                    "donor_acceptor_peak_ratio": "float",
                    "type_signal_difference": "float",
                    "score_difference_ratio": "float",
                    "signal_strength_ratio": "float",
                    "chrom": "str",
                    "window_start": "int",
                    "window_end": "int",
                    "transcript_count": "int",
                    "sequence": "str"
                },
                min_rows=1,
                max_file_size_mb=100.0  # 100MB max for analysis sequences
            ),
            
            ArtifactType.SPLICE_POSITIONS_ENHANCED: SchemaDefinition(
                artifact_type=ArtifactType.SPLICE_POSITIONS_ENHANCED,
                required_columns={
                    "gene_id", "transcript_id", "position", "predicted_position", 
                    "true_position", "strand", "chrom", "score", "context_score_p2", 
                    "context_score_m2", "context_score_p1", "context_score_m1", 
                    "relative_donor_probability", "splice_probability", 
                    "donor_acceptor_diff", "splice_neither_diff", 
                    "donor_acceptor_logodds", "splice_neither_logodds", 
                    "probability_entropy", "context_neighbor_mean", 
                    "context_asymmetry", "context_max", "donor_diff_m1", 
                    "donor_diff_m2", "donor_diff_p1", "donor_diff_p2", 
                    "donor_surge_ratio", "donor_is_local_peak", 
                    "donor_weighted_context", "donor_peak_height_ratio", 
                    "donor_second_derivative", "donor_signal_strength", 
                    "donor_context_diff_ratio", "acceptor_diff_m1", 
                    "acceptor_diff_m2", "acceptor_diff_p1", "acceptor_diff_p2", 
                    "acceptor_surge_ratio", "acceptor_is_local_peak", 
                    "acceptor_weighted_context", "acceptor_peak_height_ratio", 
                    "acceptor_second_derivative", "acceptor_signal_strength", 
                    "acceptor_context_diff_ratio", "donor_acceptor_peak_ratio", 
                    "type_signal_difference", "score_difference_ratio", 
                    "signal_strength_ratio", "donor_score", "acceptor_score", 
                    "neither_score", "pred_type", "splice_type"
                },
                optional_columns=set(),
                data_types={
                    "gene_id": "str",
                    "transcript_id": "str",
                    "position": "int",
                    "predicted_position": "int",
                    "true_position": "int",
                    "strand": "str",
                    "chrom": "str",
                    "score": "float",
                    "context_score_p2": "float",
                    "context_score_m2": "float",
                    "context_score_p1": "float",
                    "context_score_m1": "float",
                    "relative_donor_probability": "float",
                    "splice_probability": "float",
                    "donor_acceptor_diff": "float",
                    "splice_neither_diff": "float",
                    "donor_acceptor_logodds": "float",
                    "splice_neither_logodds": "float",
                    "probability_entropy": "float",
                    "context_neighbor_mean": "float",
                    "context_asymmetry": "float",
                    "context_max": "float",
                    "donor_diff_m1": "float",
                    "donor_diff_m2": "float",
                    "donor_diff_p1": "float",
                    "donor_diff_p2": "float",
                    "donor_surge_ratio": "float",
                    "donor_is_local_peak": "int",
                    "donor_weighted_context": "float",
                    "donor_peak_height_ratio": "float",
                    "donor_second_derivative": "float",
                    "donor_signal_strength": "float",
                    "donor_context_diff_ratio": "float",
                    "acceptor_diff_m1": "float",
                    "acceptor_diff_m2": "float",
                    "acceptor_diff_p1": "float",
                    "acceptor_diff_p2": "float",
                    "acceptor_surge_ratio": "float",
                    "acceptor_is_local_peak": "int",
                    "acceptor_weighted_context": "float",
                    "acceptor_peak_height_ratio": "float",
                    "acceptor_second_derivative": "float",
                    "acceptor_signal_strength": "float",
                    "acceptor_context_diff_ratio": "float",
                    "donor_acceptor_peak_ratio": "float",
                    "type_signal_difference": "float",
                    "score_difference_ratio": "float",
                    "signal_strength_ratio": "float",
                    "donor_score": "float",
                    "acceptor_score": "float",
                    "neither_score": "float",
                    "pred_type": "str",
                    "splice_type": "str"
                },
                min_rows=1,
                max_file_size_mb=200.0  # 200MB max for enhanced positions
            ),
            
            ArtifactType.SPLICE_ERRORS: SchemaDefinition(
                artifact_type=ArtifactType.SPLICE_ERRORS,
                required_columns={
                    "position", "gene_id", "strand", "window_end", 
                    "error_type", "splice_type", "transcript_id", "window_start"
                },
                optional_columns=set(),
                data_types={
                    "position": "int",
                    "gene_id": "str",
                    "strand": "str",
                    "window_end": "int",
                    "error_type": "str",
                    "splice_type": "str",
                    "transcript_id": "str",
                    "window_start": "int"
                },
                min_rows=0,  # Can be empty
                max_file_size_mb=50.0  # 50MB max for errors
            )
        }
    
    def _identify_artifact_type(self, filename: str) -> Optional[ArtifactType]:
        """Identify artifact type from filename."""
        if filename.startswith("analysis_sequences_"):
            return ArtifactType.ANALYSIS_SEQUENCES
        elif filename.startswith("splice_positions_enhanced_"):
            return ArtifactType.SPLICE_POSITIONS_ENHANCED
        elif filename.startswith("splice_errors_"):
            return ArtifactType.SPLICE_ERRORS
        return None
    
    def _get_expected_data_type(self, column: str, artifact_type: ArtifactType) -> str:
        """Get expected data type for a column."""
        schema_def = self.schema_definitions[artifact_type]
        return schema_def.data_types.get(column, "str")
    
    def _validate_data_types(self, df: pl.DataFrame, artifact_type: ArtifactType) -> Tuple[bool, List[str]]:
        """Validate data types of DataFrame columns."""
        schema_def = self.schema_definitions[artifact_type]
        errors = []
        
        for column, expected_type in schema_def.data_types.items():
            if column not in df.columns:
                continue
                
            # Check if column exists and has correct type
            col_type = str(df[column].dtype)
            
            if expected_type == "int":
                if not col_type.startswith("Int"):
                    errors.append(f"Column '{column}' should be int, got {col_type}")
            elif expected_type == "float":
                if not col_type.startswith("Float"):
                    errors.append(f"Column '{column}' should be float, got {col_type}")
            elif expected_type == "str":
                if not col_type.startswith("Utf8"):
                    errors.append(f"Column '{column}' should be str, got {col_type}")
        
        return len(errors) == 0, errors
    
    def validate_file(self, file_path: Path) -> ValidationResult:
        """
        Validate a single artifact file.
        
        Parameters
        ----------
        file_path
            Path to the artifact file to validate.
            
        Returns
        -------
        ValidationResult
            Validation result with detailed information.
        """
        file_path = Path(file_path)
        errors = []
        warnings = []
        
        # Check if file exists
        if not file_path.exists():
            return ValidationResult(
                file_path=file_path,
                artifact_type=ArtifactType.ANALYSIS_SEQUENCES,  # Placeholder
                is_valid=False,
                errors=["File does not exist"],
                warnings=[],
                row_count=0,
                column_count=0,
                file_size_mb=0.0,
                schema_matches=False,
                data_types_valid=False,
                has_required_columns=False
            )
        
        # Identify artifact type
        artifact_type = self._identify_artifact_type(file_path.name)
        if artifact_type is None:
            return ValidationResult(
                file_path=file_path,
                artifact_type=ArtifactType.ANALYSIS_SEQUENCES,  # Placeholder
                is_valid=False,
                errors=[f"Unknown artifact type for file: {file_path.name}"],
                warnings=[],
                row_count=0,
                column_count=0,
                file_size_mb=file_path.stat().st_size / (1024 * 1024),
                schema_matches=False,
                data_types_valid=False,
                has_required_columns=False
            )
        
        # Get file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Check file size limit
        schema_def = self.schema_definitions[artifact_type]
        if schema_def.max_file_size_mb and file_size_mb > schema_def.max_file_size_mb:
            warnings.append(f"File size ({file_size_mb:.1f}MB) exceeds expected maximum ({schema_def.max_file_size_mb}MB)")
        
        try:
            # Try to read the file
            df = pl.read_csv(file_path, separator="\t")
            
            # Basic statistics
            row_count = df.height
            column_count = len(df.columns)
            
            # Check minimum rows
            if row_count < schema_def.min_rows:
                errors.append(f"File has {row_count} rows, minimum expected is {schema_def.min_rows}")
            
            # Check required columns
            missing_columns = schema_def.required_columns - set(df.columns)
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check for unexpected columns
            unexpected_columns = set(df.columns) - schema_def.required_columns - schema_def.optional_columns
            if unexpected_columns:
                warnings.append(f"Unexpected columns found: {unexpected_columns}")
            
            # Validate data types
            data_types_valid, type_errors = self._validate_data_types(df, artifact_type)
            errors.extend(type_errors)
            
            # Check for empty or null values in critical columns
            if row_count > 0:
                for col in ["gene_id", "position"]:
                    if col in df.columns:
                        null_count = df[col].null_count()
                        if null_count > 0:
                            warnings.append(f"Column '{col}' has {null_count} null values")
            
            # Check for duplicate rows
            if row_count > 0:
                duplicate_count = df.height - df.unique().height
                if duplicate_count > 0:
                    warnings.append(f"File contains {duplicate_count} duplicate rows")
            
        except Exception as e:
            errors.append(f"Failed to read file: {str(e)}")
            return ValidationResult(
                file_path=file_path,
                artifact_type=artifact_type,
                is_valid=False,
                errors=errors,
                warnings=warnings,
                row_count=0,
                column_count=0,
                file_size_mb=file_size_mb,
                schema_matches=False,
                data_types_valid=False,
                has_required_columns=False
            )
        
        # Determine if file is valid
        is_valid = len(errors) == 0
        schema_matches = len(missing_columns) == 0
        has_required_columns = len(missing_columns) == 0
        
        return ValidationResult(
            file_path=file_path,
            artifact_type=artifact_type,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            row_count=row_count,
            column_count=column_count,
            file_size_mb=file_size_mb,
            schema_matches=schema_matches,
            data_types_valid=data_types_valid,
            has_required_columns=has_required_columns
        )
    
    def validate_directory(self, directory: Optional[Path] = None) -> Dict[str, List[ValidationResult]]:
        """
        Validate all artifacts in a directory.
        
        Parameters
        ----------
        directory
            Directory to validate. If None, uses self.artifacts_dir.
            
        Returns
        -------
        Dict[str, List[ValidationResult]]
            Dictionary mapping artifact types to validation results.
        """
        if directory is None:
            directory = self.artifacts_dir
        
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        
        results = {
            "valid": [],
            "invalid": [],
            "corrupted": [],
            "wrong_schema": []
        }
        
        # Find all TSV files
        tsv_files = list(directory.glob("*.tsv"))
        
        print(f"Found {len(tsv_files)} TSV files to validate...")
        
        for file_path in tsv_files:
            result = self.validate_file(file_path)
            
            if result.is_valid:
                results["valid"].append(result)
            else:
                results["invalid"].append(result)
                
                # Categorize invalid files
                if "Failed to read file" in result.errors[0]:
                    results["corrupted"].append(result)
                elif not result.has_required_columns:
                    results["wrong_schema"].append(result)
                else:
                    results["invalid"].append(result)
        
        return results
    
    def generate_report(self, results: Dict[str, List[ValidationResult]], output_file: Optional[Path] = None) -> str:
        """
        Generate a detailed validation report.
        
        Parameters
        ----------
        results
            Validation results from validate_directory().
        output_file
            Optional file to save the report to.
            
        Returns
        -------
        str
            Formatted report string.
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ARTIFACT VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        total_files = sum(len(results[key]) for key in ["valid", "invalid", "corrupted", "wrong_schema"])
        valid_files = len(results["valid"])
        invalid_files = len(results["invalid"])
        corrupted_files = len(results["corrupted"])
        wrong_schema_files = len(results["wrong_schema"])
        
        report_lines.append(f"SUMMARY:")
        report_lines.append(f"  Total files: {total_files}")
        report_lines.append(f"  Valid files: {valid_files}")
        report_lines.append(f"  Invalid files: {invalid_files}")
        report_lines.append(f"  Corrupted files: {corrupted_files}")
        report_lines.append(f"  Wrong schema files: {wrong_schema_files}")
        report_lines.append("")
        
        # Detailed breakdown by artifact type
        for category in ["valid", "invalid", "corrupted", "wrong_schema"]:
            if results[category]:
                report_lines.append(f"{category.upper()} FILES:")
                report_lines.append("-" * 40)
                
                # Group by artifact type
                by_type = {}
                for result in results[category]:
                    artifact_type = result.artifact_type.value
                    if artifact_type not in by_type:
                        by_type[artifact_type] = []
                    by_type[artifact_type].append(result)
                
                for artifact_type, type_results in by_type.items():
                    report_lines.append(f"  {artifact_type}: {len(type_results)} files")
                    for result in type_results:
                        report_lines.append(f"    - {result.file_path.name}")
                        report_lines.append(f"      Size: {result.file_size_mb:.1f}MB, Rows: {result.row_count:,}")
                        if result.errors:
                            report_lines.append(f"      Errors: {', '.join(result.errors)}")
                        if result.warnings:
                            report_lines.append(f"      Warnings: {', '.join(result.warnings)}")
                        report_lines.append("")
                
                report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 40)
        
        if corrupted_files > 0:
            report_lines.append(f"  • Delete {corrupted_files} corrupted files (cannot be read)")
        
        if wrong_schema_files > 0:
            report_lines.append(f"  • Review {wrong_schema_files} files with wrong schemas")
        
        if invalid_files > 0:
            report_lines.append(f"  • Fix {invalid_files} files with validation errors")
        
        if valid_files == total_files:
            report_lines.append("  • All files are valid! No action needed.")
        
        report_lines.append("")
        
        # File lists for easy cleanup
        if corrupted_files > 0:
            report_lines.append("CORRUPTED FILES TO DELETE:")
            report_lines.append("-" * 40)
            for result in results["corrupted"]:
                report_lines.append(f"  {result.file_path}")
            report_lines.append("")
        
        if wrong_schema_files > 0:
            report_lines.append("FILES WITH WRONG SCHEMA:")
            report_lines.append("-" * 40)
            for result in results["wrong_schema"]:
                report_lines.append(f"  {result.file_path}")
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_file}")
        
        return report


def main():
    """Command-line interface for artifact validation."""
    parser = argparse.ArgumentParser(
        description="Validate meta-model training artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Directory containing artifacts (default: data/ensembl/spliceai_eval/meta_models)"
    )
    
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Output file for validation report"
    )
    
    parser.add_argument(
        "--json-report",
        type=str,
        default=None,
        help="Output JSON file with detailed validation results"
    )
    
    parser.add_argument(
        "--list-corrupted",
        action="store_true",
        help="Only list corrupted files (for easy deletion)"
    )
    
    parser.add_argument(
        "--list-wrong-schema",
        action="store_true",
        help="Only list files with wrong schemas"
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else None
    validator = ArtifactValidator(artifacts_dir)
    
    print(f"Validating artifacts in: {validator.artifacts_dir}")
    print("=" * 60)
    
    # Run validation
    results = validator.validate_directory()
    
    # Generate report
    report = validator.generate_report(results, args.output_report)
    
    # Print report
    print(report)
    
    # Handle special output modes
    if args.list_corrupted:
        print("\nCORRUPTED FILES:")
        for result in results["corrupted"]:
            print(f"  {result.file_path}")
        return
    
    if args.list_wrong_schema:
        print("\nWRONG SCHEMA FILES:")
        for result in results["wrong_schema"]:
            print(f"  {result.file_path}")
        return
    
    # JSON output
    if args.json_report:
        json_data = {
            "summary": {
                "total_files": sum(len(results[key]) for key in ["valid", "invalid", "corrupted", "wrong_schema"]),
                "valid_files": len(results["valid"]),
                "invalid_files": len(results["invalid"]),
                "corrupted_files": len(results["corrupted"]),
                "wrong_schema_files": len(results["wrong_schema"])
            },
            "files": {
                "valid": [{"file": str(r.file_path), "size_mb": r.file_size_mb, "rows": r.row_count} for r in results["valid"]],
                "corrupted": [{"file": str(r.file_path), "errors": r.errors} for r in results["corrupted"]],
                "wrong_schema": [{"file": str(r.file_path), "errors": r.errors} for r in results["wrong_schema"]],
                "invalid": [{"file": str(r.file_path), "errors": r.errors, "warnings": r.warnings} for r in results["invalid"]]
            }
        }
        
        with open(args.json_report, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON report saved to: {args.json_report}")


if __name__ == "__main__":
    main() 