"""
Disease validation workflow.

Comprehensive workflow for validating meta-model performance on disease-specific
splice mutations from curated databases like SpliceVarDB and MutSpliceDB.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import json
import logging
from dataclasses import dataclass

# Import the existing meta-model infrastructure
from meta_spliceai.splice_engine.meta_models.builder import preprocessing
from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import run_enhanced_splice_prediction_workflow

# Import case study components
from ..data_sources import SpliceVarDBIngester, MutSpliceDBIngester, IngestionResult
from ..data_sources.base import SpliceMutation, SpliceEventType, ClinicalSignificance
from .cryptic_site_detector import CrypticSiteDetector


@dataclass
class ValidationResult:
    """Results of disease validation analysis."""
    
    # Basic metrics
    total_mutations: int
    predicted_correctly: int
    accuracy: float
    
    # Per-event-type metrics
    event_type_metrics: Dict[str, Dict[str, float]]
    
    # Clinical significance metrics
    pathogenic_accuracy: float
    benign_accuracy: float
    
    # Base vs meta comparison
    base_accuracy: float
    meta_accuracy: float
    improvement: float
    
    # Detailed results
    mutation_predictions: List[Dict[str, Any]]
    
    # Error analysis
    false_positives: List[SpliceMutation]
    false_negatives: List[SpliceMutation]


class DiseaseValidationWorkflow:
    """Workflow for validating meta-model on disease-specific mutations."""
    
    def __init__(self, work_dir: Path, meta_model_path: Optional[Path] = None):
        """
        Initialize disease validation workflow.
        
        Parameters
        ----------
        work_dir : Path
            Working directory for storing results
        meta_model_path : Path, optional
            Path to trained meta-model (if None, will train new model)
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.meta_model_path = meta_model_path
        self.logger = self._setup_logging()
        
        # Results storage
        self.validation_results = {}
        self.ingestion_results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the workflow."""
        logger = logging.getLogger("DiseaseValidationWorkflow")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.work_dir / "disease_validation.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        if not logger.handlers:
            logger.addHandler(file_handler)
        
        return logger
    
    def ingest_splice_databases(self, 
                              include_splicevardb: bool = True,
                              include_mutsplicedb: bool = True,
                              force_refresh: bool = False) -> Dict[str, IngestionResult]:
        """
        Ingest data from splice mutation databases.
        
        Parameters
        ----------
        include_splicevardb : bool
            Whether to ingest SpliceVarDB data
        include_mutsplicedb : bool
            Whether to ingest MutSpliceDB data
        force_refresh : bool
            Whether to redownload data
            
        Returns
        -------
        Dict[str, IngestionResult]
            Ingestion results keyed by database name
        """
        results = {}
        
        if include_splicevardb:
            self.logger.info("Ingesting SpliceVarDB data...")
            splicevardb_dir = self.work_dir / "splicevardb"
            ingester = SpliceVarDBIngester(splicevardb_dir)
            
            try:
                result = ingester.ingest(force_refresh=force_refresh)
                results["SpliceVarDB"] = result
                self.logger.info(f"SpliceVarDB ingestion completed: {len(result.mutations)} mutations")
            except Exception as e:
                self.logger.error(f"SpliceVarDB ingestion failed: {e}")
        
        if include_mutsplicedb:
            self.logger.info("Ingesting MutSpliceDB data...")
            mutsplicedb_dir = self.work_dir / "mutsplicedb"
            ingester = MutSpliceDBIngester(mutsplicedb_dir)
            
            try:
                result = ingester.ingest(force_refresh=force_refresh)
                results["MutSpliceDB"] = result
                self.logger.info(f"MutSpliceDB ingestion completed: {len(result.mutations)} mutations")
            except Exception as e:
                self.logger.error(f"MutSpliceDB ingestion failed: {e}")
        
        self.ingestion_results = results
        return results
    
    def create_validation_sequences(self, mutations: List[SpliceMutation], 
                                  sequence_length: int = 10000,
                                  reference_genome: str = "hg38") -> pd.DataFrame:
        """
        Create genomic sequences for splice prediction validation.
        
        Parameters
        ----------
        mutations : List[SpliceMutation]
            Mutations to create sequences for
        sequence_length : int
            Length of sequence to extract around each mutation
        reference_genome : str
            Reference genome version
            
        Returns
        -------
        pd.DataFrame
            DataFrame with sequences and metadata for validation
        """
        self.logger.info(f"Creating validation sequences for {len(mutations)} mutations...")
        
        validation_data = []
        
        for i, mutation in enumerate(mutations):
            try:
                # Calculate sequence coordinates
                start_pos = max(1, mutation.position - sequence_length // 2)
                end_pos = mutation.position + sequence_length // 2
                
                # Create sequence identifier
                sequence_id = f"{mutation.source_database}_{mutation.source_id}_{i}"
                
                # Note: In a real implementation, you would extract the actual sequence
                # using a reference genome (e.g., pyfaidx, pygenome, or similar)
                # For now, we'll create a placeholder
                sequence = "N" * sequence_length  # Placeholder sequence
                
                validation_record = {
                    "sequence_id": sequence_id,
                    "chrom": mutation.chrom,
                    "start": start_pos,
                    "end": end_pos,
                    "sequence": sequence,
                    "mutation_position": sequence_length // 2,  # Center of sequence
                    "ref_allele": mutation.ref_allele,
                    "alt_allele": mutation.alt_allele,
                    "gene_symbol": mutation.gene_symbol,
                    "gene_id": mutation.gene_id,
                    "transcript_id": mutation.transcript_id,
                    "splice_event_type": mutation.splice_event_type.value,
                    "clinical_significance": mutation.clinical_significance.value if mutation.clinical_significance else None,
                    "experimentally_validated": mutation.experimentally_validated,
                    "source_database": mutation.source_database,
                    "source_id": mutation.source_id,
                    "metadata": json.dumps(mutation.metadata) if mutation.metadata else None
                }
                
                validation_data.append(validation_record)
                
            except Exception as e:
                self.logger.warning(f"Failed to create sequence for mutation {mutation.source_id}: {e}")
                continue
        
        df = pd.DataFrame(validation_data)
        
        # Save validation sequences
        sequences_file = self.work_dir / "validation_sequences.tsv"
        df.to_csv(sequences_file, sep='\t', index=False)
        self.logger.info(f"Saved {len(df)} validation sequences to {sequences_file}")
        
        return df
    
    def detect_cryptic_sites(self, validation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect cryptic splice sites for validation sequences.
        
        Parameters
        ----------
        validation_df : pd.DataFrame
            Validation sequences with mutation information
            
        Returns
        -------
        pd.DataFrame
            DataFrame with cryptic site predictions
        """
        self.logger.info("Detecting cryptic splice sites...")
        
        # Initialize cryptic site detector
        detector = CrypticSiteDetector()
        cryptic_results = []
        
        for _, row in validation_df.iterrows():
            # Skip if no alternate sequence
            if pd.isna(row.get('alt_sequence')):
                continue
                
            # Detect cryptic sites
            cryptic_sites = detector.detect_cryptic_sites(
                reference_seq=row.get('sequence', ''),
                alternate_seq=row.get('alt_sequence', ''),
                mutation_pos=row.get('mutation_pos', 0)
            )
            
            # Add results for each detected site
            for site in cryptic_sites:
                result = {
                    'sequence_id': row['sequence_id'],
                    'mutation_id': row.get('mutation_id'),
                    'site_type': site.site_type.value,
                    'position': site.position,
                    'score': site.score,
                    'strength_change': site.strength_change,
                    'confidence': site.confidence.value,
                    'distance_from_mutation': site.distance_from_mutation
                }
                cryptic_results.append(result)
        
        cryptic_df = pd.DataFrame(cryptic_results)
        
        # Save cryptic site predictions
        cryptic_file = self.work_dir / "cryptic_sites.tsv"
        cryptic_df.to_csv(cryptic_file, sep='\t', index=False)
        self.logger.info(f"Detected {len(cryptic_df)} cryptic sites, saved to {cryptic_file}")
        
        return cryptic_df
    
    def run_splice_prediction_workflow(self, validation_df: pd.DataFrame) -> Path:
        """
        Run the splice prediction workflow on validation sequences.
        
        Parameters
        ----------
        validation_df : pd.DataFrame
            Validation sequences with metadata
            
        Returns
        -------
        Path
            Path to prediction results
        """
        self.logger.info("Running splice prediction workflow on validation data...")
        
        # Detect cryptic sites first
        cryptic_sites_df = self.detect_cryptic_sites(validation_df)
        
        # Create input for splice prediction workflow
        prediction_input = self.work_dir / "prediction_input"
        prediction_input.mkdir(exist_ok=True)
        
        # Convert validation data to format expected by splice prediction workflow
        input_sequences = []
        for _, row in validation_df.iterrows():
            seq_record = {
                "id": row["sequence_id"],
                "sequence": row["sequence"],
                "chrom": row["chrom"],
                "start": row["start"],
                "end": row["end"]
            }
            input_sequences.append(seq_record)
        
        # Save sequences in expected format
        sequences_file = prediction_input / "sequences.json"
        with open(sequences_file, 'w') as f:
            json.dump(input_sequences, f, indent=2)
        
        # Initialize and run splice prediction workflow
        prediction_output = self.work_dir / "splice_predictions"
        workflow = SplicePredictionWorkflow()
        
        try:
            # Run base model predictions (SpliceAI)
            base_results = workflow.run_base_predictions(
                sequences_file=sequences_file,
                output_dir=prediction_output / "base_predictions"
            )
            
            # Run meta-model predictions if model is available
            if self.meta_model_path and self.meta_model_path.exists():
                meta_results = workflow.run_meta_predictions(
                    base_results=base_results,
                    meta_model_path=self.meta_model_path,
                    output_dir=prediction_output / "meta_predictions"
                )
            else:
                self.logger.warning("Meta-model not available, running base predictions only")
                meta_results = base_results
            
            self.logger.info(f"Splice prediction completed. Results saved to {prediction_output}")
            return prediction_output
            
        except Exception as e:
            self.logger.error(f"Splice prediction workflow failed: {e}")
            raise
    
    def evaluate_predictions(self, 
                           validation_df: pd.DataFrame,
                           prediction_results_dir: Path) -> ValidationResult:
        """
        Evaluate predictions against known mutation effects.
        
        Parameters
        ----------
        validation_df : pd.DataFrame
            Original validation data with ground truth
        prediction_results_dir : Path
            Directory containing prediction results
            
        Returns
        -------
        ValidationResult
            Comprehensive evaluation results
        """
        self.logger.info("Evaluating predictions against known mutation effects...")
        
        # Load prediction results
        base_predictions_file = prediction_results_dir / "base_predictions" / "predictions.tsv"
        meta_predictions_file = prediction_results_dir / "meta_predictions" / "predictions.tsv"
        
        if not base_predictions_file.exists():
            raise FileNotFoundError(f"Base predictions not found: {base_predictions_file}")
        
        base_df = pd.read_csv(base_predictions_file, sep='\t')
        
        if meta_predictions_file.exists():
            meta_df = pd.read_csv(meta_predictions_file, sep='\t')
        else:
            self.logger.warning("Meta predictions not found, using base predictions only")
            meta_df = base_df.copy()
        
        # Merge with validation data
        merged_df = validation_df.merge(
            base_df, left_on="sequence_id", right_on="sequence_id", how="inner", suffixes=("", "_base")
        )
        merged_df = merged_df.merge(
            meta_df, left_on="sequence_id", right_on="sequence_id", how="inner", suffixes=("", "_meta")
        )
        
        # Evaluate predictions
        mutation_predictions = []
        correct_base = 0
        correct_meta = 0
        false_positives = []
        false_negatives = []
        
        # Event type metrics
        event_type_metrics = {}
        
        # Clinical significance metrics
        pathogenic_correct = 0
        pathogenic_total = 0
        benign_correct = 0
        benign_total = 0
        
        for _, row in merged_df.iterrows():
            # Determine ground truth
            expected_effect = row["splice_event_type"]
            clinical_sig = row["clinical_significance"]
            experimentally_validated = row["experimentally_validated"]
            
            # Extract predictions (placeholder logic - adapt based on actual prediction format)
            base_prediction = self._interpret_base_prediction(row)
            meta_prediction = self._interpret_meta_prediction(row)
            
            # Evaluate correctness
            base_correct = self._is_prediction_correct(expected_effect, base_prediction)
            meta_correct = self._is_prediction_correct(expected_effect, meta_prediction)
            
            if base_correct:
                correct_base += 1
            if meta_correct:
                correct_meta += 1
            
            # Track clinical significance accuracy
            if clinical_sig in ["pathogenic", "likely_pathogenic"]:
                pathogenic_total += 1
                if meta_correct:
                    pathogenic_correct += 1
            elif clinical_sig in ["benign", "likely_benign"]:
                benign_total += 1
                if meta_correct:
                    benign_correct += 1
            
            # Track event type metrics
            if expected_effect not in event_type_metrics:
                event_type_metrics[expected_effect] = {"total": 0, "correct": 0}
            event_type_metrics[expected_effect]["total"] += 1
            if meta_correct:
                event_type_metrics[expected_effect]["correct"] += 1
            
            # Store detailed prediction
            mutation_pred = {
                "sequence_id": row["sequence_id"],
                "gene_symbol": row["gene_symbol"],
                "mutation": f"{row['ref_allele']}>{row['alt_allele']}",
                "expected_effect": expected_effect,
                "base_prediction": base_prediction,
                "meta_prediction": meta_prediction,
                "base_correct": base_correct,
                "meta_correct": meta_correct,
                "clinical_significance": clinical_sig,
                "experimentally_validated": experimentally_validated,
                "source_database": row["source_database"]
            }
            mutation_predictions.append(mutation_pred)
            
            # Track errors for analysis
            if not meta_correct:
                # Create SpliceMutation object for error tracking
                mutation = self._create_mutation_from_row(row)
                if self._is_false_positive(expected_effect, meta_prediction):
                    false_positives.append(mutation)
                else:
                    false_negatives.append(mutation)
        
        # Calculate final metrics
        total_mutations = len(merged_df)
        base_accuracy = correct_base / total_mutations if total_mutations > 0 else 0.0
        meta_accuracy = correct_meta / total_mutations if total_mutations > 0 else 0.0
        improvement = meta_accuracy - base_accuracy
        
        pathogenic_accuracy = pathogenic_correct / pathogenic_total if pathogenic_total > 0 else 0.0
        benign_accuracy = benign_correct / benign_total if benign_total > 0 else 0.0
        
        # Calculate event type accuracies
        for event_type in event_type_metrics:
            metrics = event_type_metrics[event_type]
            metrics["accuracy"] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0.0
        
        # Create validation result
        result = ValidationResult(
            total_mutations=total_mutations,
            predicted_correctly=correct_meta,
            accuracy=meta_accuracy,
            event_type_metrics=event_type_metrics,
            pathogenic_accuracy=pathogenic_accuracy,
            benign_accuracy=benign_accuracy,
            base_accuracy=base_accuracy,
            meta_accuracy=meta_accuracy,
            improvement=improvement,
            mutation_predictions=mutation_predictions,
            false_positives=false_positives,
            false_negatives=false_negatives
        )
        
        # Save detailed results
        self._save_validation_results(result)
        
        self.logger.info(f"Evaluation completed. Meta-model accuracy: {meta_accuracy:.3f}")
        self.logger.info(f"Improvement over base model: {improvement:+.3f}")
        
        return result
    
    def _interpret_base_prediction(self, row: pd.Series) -> str:
        """Interpret base model predictions (placeholder implementation)."""
        # This should be adapted based on actual prediction format
        # For now, return a placeholder
        return "predicted_effect"
    
    def _interpret_meta_prediction(self, row: pd.Series) -> str:
        """Interpret meta-model predictions (placeholder implementation)."""
        # This should be adapted based on actual prediction format
        # For now, return a placeholder
        return "predicted_effect"
    
    def _is_prediction_correct(self, expected: str, predicted: str) -> bool:
        """Determine if prediction matches expected effect."""
        # Implement logic to compare expected vs predicted effects
        # This is simplified - real implementation would need more sophisticated comparison
        return expected.lower() == predicted.lower()
    
    def _is_false_positive(self, expected: str, predicted: str) -> bool:
        """Determine if this is a false positive."""
        # Simplified logic - adapt based on your specific needs
        return expected == "no_effect" and predicted != "no_effect"
    
    def _create_mutation_from_row(self, row: pd.Series) -> SpliceMutation:
        """Create SpliceMutation object from DataFrame row."""
        return SpliceMutation(
            chrom=row["chrom"],
            position=row.get("mutation_position", 0),
            ref_allele=row["ref_allele"],
            alt_allele=row["alt_allele"],
            gene_id=row["gene_id"],
            gene_symbol=row["gene_symbol"],
            transcript_id=row.get("transcript_id", ""),
            splice_event_type=SpliceEventType(row["splice_event_type"]),
            affected_site_type="unknown",
            clinical_significance=ClinicalSignificance(row["clinical_significance"]) if row["clinical_significance"] else None,
            experimentally_validated=row["experimentally_validated"],
            source_database=row["source_database"],
            source_id=row["source_id"]
        )
    
    def _save_validation_results(self, result: ValidationResult) -> None:
        """Save validation results to files."""
        results_dir = self.work_dir / "validation_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save summary metrics
        summary = {
            "total_mutations": result.total_mutations,
            "base_accuracy": result.base_accuracy,
            "meta_accuracy": result.meta_accuracy,
            "improvement": result.improvement,
            "pathogenic_accuracy": result.pathogenic_accuracy,
            "benign_accuracy": result.benign_accuracy,
            "event_type_metrics": result.event_type_metrics
        }
        
        with open(results_dir / "validation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed predictions
        predictions_df = pd.DataFrame(result.mutation_predictions)
        predictions_df.to_csv(results_dir / "detailed_predictions.tsv", sep='\t', index=False)
        
        # Save error analysis
        if result.false_positives:
            fp_data = [
                {
                    "gene_symbol": m.gene_symbol,
                    "mutation": f"{m.ref_allele}>{m.alt_allele}",
                    "expected_effect": m.splice_event_type.value,
                    "source": m.source_database
                }
                for m in result.false_positives
            ]
            fp_df = pd.DataFrame(fp_data)
            fp_df.to_csv(results_dir / "false_positives.tsv", sep='\t', index=False)
        
        if result.false_negatives:
            fn_data = [
                {
                    "gene_symbol": m.gene_symbol,
                    "mutation": f"{m.ref_allele}>{m.alt_allele}",
                    "expected_effect": m.splice_event_type.value,
                    "source": m.source_database
                }
                for m in result.false_negatives
            ]
            fn_df = pd.DataFrame(fn_data)
            fn_df.to_csv(results_dir / "false_negatives.tsv", sep='\t', index=False)
    
    def run_disease_specific_validation(self,
                                      diseases: List[str],
                                      databases: List[str] = ["SpliceVarDB", "MutSpliceDB"],
                                      min_rna_evidence: int = 5) -> Dict[str, ValidationResult]:
        """
        Run comprehensive validation for specific diseases.
        
        Parameters
        ----------
        diseases : List[str]
            Disease names to focus on
        databases : List[str]
            Databases to use for validation
        min_rna_evidence : int
            Minimum RNA evidence threshold
            
        Returns
        -------
        Dict[str, ValidationResult]
            Validation results keyed by disease name
        """
        self.logger.info(f"Running disease-specific validation for: {', '.join(diseases)}")
        
        # Ingest database data
        ingestion_results = self.ingest_splice_databases(
            include_splicevardb="SpliceVarDB" in databases,
            include_mutsplicedb="MutSpliceDB" in databases
        )
        
        disease_results = {}
        
        for disease in diseases:
            self.logger.info(f"Processing disease: {disease}")
            
            # Collect mutations for this disease
            disease_mutations = []
            
            for db_name, ingestion_result in ingestion_results.items():
                for mutation in ingestion_result.mutations:
                    # Check if mutation is related to this disease
                    if (mutation.disease_context and 
                        disease.lower() in mutation.disease_context.lower()):
                        
                        # Apply quality filters
                        if mutation.experimentally_validated:
                            if (mutation.metadata and 
                                mutation.metadata.get('rna_support_reads', 0) >= min_rna_evidence):
                                disease_mutations.append(mutation)
                            elif not mutation.metadata.get('rna_support_reads'):
                                # Include if no read count but marked as validated
                                disease_mutations.append(mutation)
            
            if not disease_mutations:
                self.logger.warning(f"No validated mutations found for {disease}")
                continue
            
            self.logger.info(f"Found {len(disease_mutations)} validated mutations for {disease}")
            
            # Create validation sequences
            validation_df = self.create_validation_sequences(disease_mutations)
            
            # Run splice prediction workflow
            prediction_results = self.run_splice_prediction_workflow(validation_df)
            
            # Evaluate predictions
            validation_result = self.evaluate_predictions(validation_df, prediction_results)
            
            disease_results[disease] = validation_result
            
            self.logger.info(f"Disease {disease} validation completed. Accuracy: {validation_result.accuracy:.3f}")
        
        # Save combined results
        self._save_disease_comparison(disease_results)
        
        return disease_results
    
    def _save_disease_comparison(self, disease_results: Dict[str, ValidationResult]) -> None:
        """Save comparison across diseases."""
        comparison_data = []
        
        for disease, result in disease_results.items():
            comparison_data.append({
                "disease": disease,
                "total_mutations": result.total_mutations,
                "base_accuracy": result.base_accuracy,
                "meta_accuracy": result.meta_accuracy,
                "improvement": result.improvement,
                "pathogenic_accuracy": result.pathogenic_accuracy,
                "benign_accuracy": result.benign_accuracy
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(self.work_dir / "disease_comparison.tsv", sep='\t', index=False)
        
        self.logger.info(f"Disease comparison saved to {self.work_dir / 'disease_comparison.tsv'}")


# Example usage function
def run_met_exon14_case_study(work_dir: Path, meta_model_path: Optional[Path] = None) -> ValidationResult:
    """
    Example: Run MET exon 14 skipping case study.
    
    This demonstrates validation on the clinically relevant MET exon 14 skipping
    mutations that are targetable with capmatinib and tepotinib.
    """
    workflow = DiseaseValidationWorkflow(work_dir, meta_model_path)
    
    # Focus on lung cancer mutations
    results = workflow.run_disease_specific_validation(
        diseases=["lung_cancer", "NSCLC", "LUAD"],
        databases=["MutSpliceDB"],  # TCGA data has good lung cancer representation
        min_rna_evidence=10  # High quality threshold
    )
    
    return results.get("lung_cancer", results.get("NSCLC", results.get("LUAD"))) 