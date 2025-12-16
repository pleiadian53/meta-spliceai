#!/usr/bin/env python3
"""
Schema Adapters for Multi-Model Integration

This module provides a systematic approach to handling different splice site annotation
schemas from various base models (SpliceAI, OpenSpliceAI, future models) and converting
them to the canonical MetaSpliceAI format.

The adapter pattern ensures:
- Centralized schema management
- Consistent conversion logic
- Easy addition of new base models
- Better error handling and validation
- Separation of concerns (extraction vs adaptation)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from dataclasses import dataclass
from enum import Enum

class SpliceModelType(Enum):
    """Enumeration of supported splice prediction models."""
    METASPLICEAI = "splicesurveyor"
    SPLICEAI = "spliceai"
    OPENSPLICEAI = "openspliceai"
    ALIGNED_EXTRACTOR = "aligned_extractor"

@dataclass
class SpliceAnnotationSchema:
    """
    Defines the schema for splice site annotations.
    
    This serves as the canonical format that all models should be adapted to.
    """
    # Column names
    chromosome_col: str = "chrom"
    start_col: str = "start"
    end_col: str = "end"
    strand_col: str = "strand"
    site_type_col: str = "site_type"
    gene_id_col: str = "gene_id"
    transcript_id_col: str = "transcript_id"
    
    # Optional columns
    score_col: Optional[str] = None
    sequence_col: Optional[str] = None
    
    # Data types
    dtypes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.dtypes is None:
            self.dtypes = {
                self.chromosome_col: "str",
                self.start_col: "int64",
                self.end_col: "int64",
                self.strand_col: "str",
                self.site_type_col: "str",
                self.gene_id_col: "str",
                self.transcript_id_col: "str"
            }
    
    @property
    def required_columns(self) -> List[str]:
        """Get list of required columns."""
        return [
            self.chromosome_col,
            self.start_col,
            self.end_col,
            self.strand_col,
            self.site_type_col,
            self.gene_id_col,
            self.transcript_id_col
        ]
    
    @property
    def optional_columns(self) -> List[str]:
        """Get list of optional columns."""
        optional = []
        if self.score_col:
            optional.append(self.score_col)
        if self.sequence_col:
            optional.append(self.sequence_col)
        return optional

# Canonical MetaSpliceAI schema
METASPLICEAI_SCHEMA = SpliceAnnotationSchema()

class SpliceSchemaAdapter(ABC):
    """
    Abstract base class for splice annotation schema adapters.
    
    Each base model (SpliceAI, OpenSpliceAI, etc.) should have its own adapter
    that knows how to convert from that model's native format to the canonical
    MetaSpliceAI format.
    """
    
    def __init__(self, model_type: SpliceModelType, target_schema: SpliceAnnotationSchema = None):
        self.model_type = model_type
        self.target_schema = target_schema or METASPLICEAI_SCHEMA
    
    @abstractmethod
    def get_source_schema(self) -> Dict[str, str]:
        """
        Get the source schema mapping for this model.
        
        Returns:
            Dict mapping canonical column names to source column names
        """
        pass
    
    @abstractmethod
    def get_column_mapping(self) -> Dict[str, str]:
        """
        Get column name mapping from source to target format.
        
        Returns:
            Dict mapping source column names to target column names
        """
        pass
    
    def adapt_dataframe(self, df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """
        Adapt a DataFrame from source schema to target schema.
        
        Args:
            df: Source DataFrame
            validate: Whether to validate the result
            
        Returns:
            Adapted DataFrame in target schema format
        """
        if df is None or df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        adapted_df = df.copy()
        
        # Apply column mapping
        column_mapping = self.get_column_mapping()
        adapted_df = adapted_df.rename(columns=column_mapping)
        
        # Apply custom transformations
        adapted_df = self._apply_custom_transformations(adapted_df)
        
        # Ensure required columns exist
        adapted_df = self._ensure_required_columns(adapted_df)
        
        # Remove extra columns not in target schema
        adapted_df = self._filter_to_target_columns(adapted_df)
        
        # Reorder columns to match target schema
        adapted_df = self._reorder_columns(adapted_df)
        
        # Apply data type conversions
        adapted_df = self._apply_data_types(adapted_df)
        
        if validate:
            self._validate_adapted_dataframe(adapted_df)
        
        return adapted_df
    
    def _apply_custom_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply model-specific transformations.
        
        Override in subclasses for model-specific logic.
        """
        return df
    
    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns are present."""
        for col in self.target_schema.required_columns:
            if col not in df.columns:
                # Try to derive missing columns
                df = self._derive_missing_column(df, col)
        
        return df
    
    def _derive_missing_column(self, df: pd.DataFrame, missing_col: str) -> pd.DataFrame:
        """
        Attempt to derive missing columns from existing data.
        
        Override in subclasses for model-specific derivation logic.
        """
        if missing_col == self.target_schema.end_col and self.target_schema.start_col in df.columns:
            # For splice sites, end is typically same as start
            df[missing_col] = df[self.target_schema.start_col]
        
        return df
    
    def _filter_to_target_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns not in target schema."""
        target_columns = self.target_schema.required_columns + self.target_schema.optional_columns
        existing_target_columns = [col for col in target_columns if col in df.columns]
        return df[existing_target_columns]
    
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns to match target schema."""
        ordered_columns = []
        
        # Add required columns first
        for col in self.target_schema.required_columns:
            if col in df.columns:
                ordered_columns.append(col)
        
        # Add optional columns
        for col in self.target_schema.optional_columns:
            if col in df.columns:
                ordered_columns.append(col)
        
        return df[ordered_columns]
    
    def _apply_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply target schema data types."""
        for col, dtype in self.target_schema.dtypes.items():
            if col in df.columns:
                try:
                    if dtype == "int64":
                        df[col] = df[col].astype("int64")
                    elif dtype == "str":
                        df[col] = df[col].astype("str")
                    # Add more type conversions as needed
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to {dtype}: {e}")
        
        return df
    
    def _validate_adapted_dataframe(self, df: pd.DataFrame):
        """Validate that the adapted DataFrame meets target schema requirements."""
        # Check required columns
        missing_required = [col for col in self.target_schema.required_columns if col not in df.columns]
        if missing_required:
            raise ValueError(f"Missing required columns after adaptation: {missing_required}")
        
        # Check data types
        for col, expected_dtype in self.target_schema.dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if expected_dtype == "int64" and not actual_dtype.startswith("int"):
                    print(f"Warning: Column {col} has type {actual_dtype}, expected {expected_dtype}")

class AlignedExtractorAdapter(SpliceSchemaAdapter):
    """Adapter for AlignedSpliceExtractor output format."""
    
    def __init__(self):
        super().__init__(SpliceModelType.ALIGNED_EXTRACTOR)
    
    def get_source_schema(self) -> Dict[str, str]:
        """AlignedSpliceExtractor schema mapping."""
        return {
            "chromosome_col": "chromosome",
            "start_col": "position",
            "end_col": "position",  # Will be derived
            "strand_col": "strand",
            "site_type_col": "splice_type",
            "gene_id_col": "gene_id",
            "transcript_id_col": "transcript_id"
        }
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Column mapping from AlignedSpliceExtractor to MetaSpliceAI."""
        return {
            "chromosome": "chrom",
            "position": "start",
            "splice_type": "site_type"
            # gene_id, transcript_id, strand remain the same
        }
    
    def _apply_custom_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply AlignedSpliceExtractor-specific transformations."""
        # Remove coordinate_system column if present
        if "coordinate_system" in df.columns:
            df = df.drop("coordinate_system", axis=1)
        
        return df

class SpliceAIAdapter(SpliceSchemaAdapter):
    """Adapter for SpliceAI output format."""
    
    def __init__(self):
        super().__init__(SpliceModelType.SPLICEAI)
    
    def get_source_schema(self) -> Dict[str, str]:
        """SpliceAI schema mapping."""
        # TODO: Define based on actual SpliceAI output format
        return {
            "chromosome_col": "chrom",
            "start_col": "pos",
            "end_col": "pos",
            "strand_col": "strand",
            "site_type_col": "type",
            "gene_id_col": "gene",
            "transcript_id_col": "transcript"
        }
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Column mapping from SpliceAI to MetaSpliceAI."""
        return {
            "pos": "start",
            "type": "site_type",
            "gene": "gene_id",
            "transcript": "transcript_id"
        }

class OpenSpliceAIAdapter(SpliceSchemaAdapter):
    """Adapter for OpenSpliceAI output format."""
    
    def __init__(self):
        super().__init__(SpliceModelType.OPENSPLICEAI)
    
    def get_source_schema(self) -> Dict[str, str]:
        """OpenSpliceAI schema mapping."""
        # TODO: Define based on actual OpenSpliceAI output format
        return {
            "chromosome_col": "chr",
            "start_col": "position",
            "end_col": "position",
            "strand_col": "strand",
            "site_type_col": "splice_type",
            "gene_id_col": "gene_id",
            "transcript_id_col": "transcript_id"
        }
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Column mapping from OpenSpliceAI to MetaSpliceAI."""
        return {
            "chr": "chrom"
            # Other columns already match
        }

# Factory function for creating adapters
def create_schema_adapter(model_type: Union[str, SpliceModelType]) -> SpliceSchemaAdapter:
    """
    Factory function to create appropriate schema adapter.
    
    Args:
        model_type: Type of model to create adapter for
        
    Returns:
        Appropriate schema adapter instance
    """
    if isinstance(model_type, str):
        model_type = SpliceModelType(model_type.lower())
    
    adapter_map = {
        SpliceModelType.ALIGNED_EXTRACTOR: AlignedExtractorAdapter,
        SpliceModelType.SPLICEAI: SpliceAIAdapter,
        SpliceModelType.OPENSPLICEAI: OpenSpliceAIAdapter,
    }
    
    if model_type not in adapter_map:
        raise ValueError(f"No adapter available for model type: {model_type}")
    
    return adapter_map[model_type]()

# Convenience function for direct adaptation
def adapt_splice_annotations(df: pd.DataFrame, 
                           source_model: Union[str, SpliceModelType],
                           validate: bool = True) -> pd.DataFrame:
    """
    Convenience function to adapt splice annotations from any supported model.
    
    Args:
        df: Source DataFrame
        source_model: Source model type
        validate: Whether to validate the result
        
    Returns:
        Adapted DataFrame in MetaSpliceAI format
    """
    adapter = create_schema_adapter(source_model)
    return adapter.adapt_dataframe(df, validate=validate)
