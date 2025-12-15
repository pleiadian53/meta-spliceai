"""
Mutation analysis workflow.

Focused workflow for analyzing individual splice-altering mutations,
their effects, and comparing base vs meta-model predictions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import json
import logging
from dataclasses import dataclass
import numpy as np

# Import the existing meta-model infrastructure
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import run_enhanced_splice_prediction_workflow

# Import case study components
from ..data_sources.base import SpliceMutation, SpliceEventType, ClinicalSignificance
from ..formats.hgvs_parser import HGVSParser
from ..analysis.splicing_pattern_analyzer import SplicingPatternAnalyzer, SpliceSite


@dataclass
class MutationAnalysisResult:
    """Results of detailed mutation analysis."""
    
    # Mutation information
    mutation: SpliceMutation
    
    # Base model predictions
    base_donor_score: float
    base_acceptor_score: float
    base_combined_score: float
    
    # Meta model predictions  
    meta_donor_score: float
    meta_acceptor_score: float
    meta_combined_score: float
    meta_prediction_class: str
    
    # Comparative metrics
    score_improvement: float
    prediction_agreement: bool
    confidence_delta: float
    
    # Context analysis
    local_sequence: str
    cryptic_sites_detected: List[Dict[str, Any]]
    splice_strength_analysis: Dict[str, float]
    
    # Validation data
    experimental_evidence: Optional[str]
    literature_support: List[str]


class MutationAnalysisWorkflow:
    """Workflow for detailed analysis of splice-altering mutations."""
    
    def __init__(self, work_dir: Path, meta_model_path: Optional[Path] = None):
        """
        Initialize mutation analysis workflow.
        
        Parameters
        ----------
        work_dir : Path
            Working directory for analysis results
        meta_model_path : Path, optional
            Path to trained meta-model
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.meta_model_path = meta_model_path
        self.logger = self._setup_logging()
        self.hgvs_parser = HGVSParser()
        
        # Analysis results storage
        self.analysis_results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the workflow."""
        logger = logging.getLogger("MutationAnalysisWorkflow")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.work_dir / "mutation_analysis.log"
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
    
    def extract_sequence_context(self, mutation: SpliceMutation, 
                                context_length: int = 300) -> str:
        """
        Extract genomic sequence around mutation.
        
        Parameters
        ----------
        mutation : SpliceMutation
            Mutation to analyze
        context_length : int
            Length of sequence context to extract
            
        Returns
        -------
        str
            Genomic sequence around mutation
        """
        # In a real implementation, this would use a reference genome
        # For now, return a placeholder sequence
        self.logger.info(f"Extracting sequence context for {mutation.gene_symbol}:{mutation.position}")
        
        # Placeholder sequence - in practice, use pyfaidx or similar
        placeholder_sequence = "N" * context_length
        
        # Mark mutation position (center of sequence)
        mut_pos = context_length // 2
        ref_allele = mutation.ref_allele
        alt_allele = mutation.alt_allele
        
        # Create wild-type and mutant sequences
        wt_sequence = (placeholder_sequence[:mut_pos] + 
                      ref_allele + 
                      placeholder_sequence[mut_pos + len(ref_allele):])
        
        return wt_sequence
    
    def run_splice_prediction(self, mutation: SpliceMutation, 
                            sequence: str) -> Dict[str, Any]:
        """
        Run splice prediction on mutation sequence.
        
        Parameters
        ----------
        mutation : SpliceMutation
            Mutation to analyze
        sequence : str
            Genomic sequence containing mutation
            
        Returns
        -------
        Dict[str, Any]
            Prediction results from base and meta models
        """
        self.logger.info(f"Running splice prediction for {mutation.source_id}")
        
        # Create input for splice prediction workflow
        prediction_input = {
            "id": f"{mutation.source_id}_analysis",
            "sequence": sequence,
            "chrom": mutation.chrom,
            "start": mutation.position - len(sequence) // 2,
            "end": mutation.position + len(sequence) // 2,
            "mutation_position": len(sequence) // 2,
            "ref_allele": mutation.ref_allele,
            "alt_allele": mutation.alt_allele
        }
        
        # Initialize splice prediction workflow
        workflow = SplicePredictionWorkflow()
        
        try:
            # Run base model prediction
            base_results = workflow.predict_splice_effects(
                sequence=sequence,
                output_dir=self.work_dir / "base_predictions"
            )
            
            # Run meta-model prediction if available
            if self.meta_model_path and self.meta_model_path.exists():
                meta_results = workflow.predict_with_meta_model(
                    base_results=base_results,
                    meta_model_path=self.meta_model_path,
                    output_dir=self.work_dir / "meta_predictions"
                )
            else:
                self.logger.warning("Meta-model not available for mutation analysis")
                meta_results = base_results
            
            return {
                "base_results": base_results,
                "meta_results": meta_results,
                "prediction_input": prediction_input
            }
            
        except Exception as e:
            self.logger.error(f"Splice prediction failed for {mutation.source_id}: {e}")
            return {}
    
    def detect_cryptic_splice_sites(self, sequence: str, 
                                  mutation_position: int) -> List[Dict[str, Any]]:
        """
        Detect potential cryptic splice sites in sequence.
        
        Parameters
        ----------
        sequence : str
            Genomic sequence to analyze
        mutation_position : int
            Position of mutation within sequence
            
        Returns
        -------
        List[Dict[str, Any]]
            List of detected cryptic splice sites
        """
        cryptic_sites = []
        
        # Common splice site motifs
        donor_motifs = ["GT", "GC"]
        acceptor_motifs = ["AG"]
        
        # Scan for donor sites
        for i, motif in enumerate(donor_motifs):
            for pos in range(len(sequence) - len(motif) + 1):
                if sequence[pos:pos + len(motif)] == motif:
                    # Calculate distance from mutation
                    distance = pos - mutation_position
                    
                    # Get surrounding context for strength estimation
                    context_start = max(0, pos - 3)
                    context_end = min(len(sequence), pos + len(motif) + 3)
                    context = sequence[context_start:context_end]
                    
                    # Placeholder strength score (in practice, use MaxEntScan or similar)
                    strength_score = np.random.uniform(0, 10)  # Placeholder
                    
                    cryptic_sites.append({
                        "type": "donor",
                        "motif": motif,
                        "position": pos,
                        "distance_from_mutation": distance,
                        "sequence_context": context,
                        "strength_score": strength_score,
                        "is_novel": abs(distance) < 50  # Consider novel if close to mutation
                    })
        
        # Scan for acceptor sites
        for motif in acceptor_motifs:
            for pos in range(len(sequence) - len(motif) + 1):
                if sequence[pos:pos + len(motif)] == motif:
                    distance = pos - mutation_position
                    
                    context_start = max(0, pos - 10)  # Larger context for acceptor
                    context_end = min(len(sequence), pos + len(motif) + 3)
                    context = sequence[context_start:context_end]
                    
                    # Check for polypyrimidine tract upstream
                    upstream = sequence[max(0, pos - 20):pos]
                    py_content = (upstream.count('T') + upstream.count('C')) / len(upstream) if upstream else 0
                    
                    strength_score = np.random.uniform(0, 10) * (py_content + 0.1)  # Placeholder
                    
                    cryptic_sites.append({
                        "type": "acceptor",
                        "motif": motif,
                        "position": pos,
                        "distance_from_mutation": distance,
                        "sequence_context": context,
                        "strength_score": strength_score,
                        "polypyrimidine_content": py_content,
                        "is_novel": abs(distance) < 50
                    })
        
        # Sort by strength score (descending)
        cryptic_sites.sort(key=lambda x: x["strength_score"], reverse=True)
        
        return cryptic_sites
    
    def analyze_splicing_patterns(self, splice_sites: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze alternative splicing patterns from delta scores.
        
        Parameters
        ----------
        splice_sites : List[Dict[str, Any]]
            List of splice sites with delta scores
            
        Returns
        -------
        Dict[str, Any]
            Analysis results including detected patterns
        """
        # Convert to SpliceSite objects for pattern analyzer
        sites = []
        for site_data in splice_sites:
            site = SpliceSite(
                chrom=site_data.get('chrom', 'chr1'),
                position=site_data.get('position', 0),
                site_type=site_data.get('site_type', 'donor'),
                delta_score=site_data.get('delta_score', 0.0),
                ref_score=site_data.get('ref_score', 0.0),
                alt_score=site_data.get('alt_score', 0.0),
                gene_id=site_data.get('gene_id'),
                transcript_id=site_data.get('transcript_id')
            )
            sites.append(site)
        
        # Initialize pattern analyzer
        analyzer = SplicingPatternAnalyzer()
        
        # Detect patterns
        patterns = analyzer.detect_patterns(sites)
        
        # Generate summary
        summary = analyzer.generate_pattern_summary(patterns)
        
        # Extract key metrics
        results = {
            'num_patterns': len(patterns),
            'patterns': patterns,
            'summary': summary,
            'pattern_types': [p.pattern_type.value for p in patterns],
            'max_confidence': max([p.confidence for p in patterns]) if patterns else 0,
            'max_severity': max([p.severity for p in patterns]) if patterns else 0
        }
        
        return results
    
    def analyze_splice_strength_changes(self, mutation: SpliceMutation,
                                      wt_sequence: str,
                                      mut_sequence: str) -> Dict[str, float]:
        """
        Analyze changes in splice site strength due to mutation.
        
        Parameters
        ----------
        mutation : SpliceMutation
            Mutation being analyzed
        wt_sequence : str
            Wild-type sequence
        mut_sequence : str
            Mutant sequence
            
        Returns
        -------
        Dict[str, float]
            Splice strength analysis results
        """
        # In practice, this would use tools like MaxEntScan, SplicePort, etc.
        # For now, provide placeholder analysis
        
        analysis = {
            "wt_donor_strength": np.random.uniform(0, 10),
            "mut_donor_strength": np.random.uniform(0, 10),
            "wt_acceptor_strength": np.random.uniform(0, 10),
            "mut_acceptor_strength": np.random.uniform(0, 10),
            "donor_strength_delta": 0.0,
            "acceptor_strength_delta": 0.0,
            "overall_strength_change": 0.0
        }
        
        # Calculate deltas
        analysis["donor_strength_delta"] = (analysis["mut_donor_strength"] - 
                                          analysis["wt_donor_strength"])
        analysis["acceptor_strength_delta"] = (analysis["mut_acceptor_strength"] - 
                                             analysis["wt_acceptor_strength"])
        analysis["overall_strength_change"] = (analysis["donor_strength_delta"] + 
                                             analysis["acceptor_strength_delta"]) / 2
        
        return analysis
    
    def analyze_single_mutation(self, mutation: SpliceMutation) -> MutationAnalysisResult:
        """
        Perform comprehensive analysis of a single mutation.
        
        Parameters
        ----------
        mutation : SpliceMutation
            Mutation to analyze
            
        Returns
        -------
        MutationAnalysisResult
            Comprehensive analysis results
        """
        self.logger.info(f"Starting comprehensive analysis of {mutation.source_id}")
        
        # Extract sequence context
        wt_sequence = self.extract_sequence_context(mutation)
        
        # Create mutant sequence
        mut_pos = len(wt_sequence) // 2
        mut_sequence = (wt_sequence[:mut_pos] + 
                       mutation.alt_allele + 
                       wt_sequence[mut_pos + len(mutation.ref_allele):])
        
        # Run splice predictions
        prediction_results = self.run_splice_prediction(mutation, wt_sequence)
        
        # Extract prediction scores (placeholder - adapt to actual prediction format)
        base_results = prediction_results.get("base_results", {})
        meta_results = prediction_results.get("meta_results", {})
        
        base_donor_score = base_results.get("donor_score", 0.0)
        base_acceptor_score = base_results.get("acceptor_score", 0.0)
        base_combined_score = base_donor_score + base_acceptor_score
        
        meta_donor_score = meta_results.get("donor_score", base_donor_score)
        meta_acceptor_score = meta_results.get("acceptor_score", base_acceptor_score)
        meta_combined_score = meta_donor_score + meta_acceptor_score
        meta_prediction_class = meta_results.get("predicted_class", "unknown")
        
        # Calculate improvement metrics
        score_improvement = meta_combined_score - base_combined_score
        prediction_agreement = abs(score_improvement) < 0.1  # Placeholder threshold
        confidence_delta = abs(meta_combined_score - 0.5) - abs(base_combined_score - 0.5)
        
        # Detect cryptic sites
        cryptic_sites = self.detect_cryptic_splice_sites(wt_sequence, mut_pos)
        
        # Analyze splice strength changes
        strength_analysis = self.analyze_splice_strength_changes(
            mutation, wt_sequence, mut_sequence
        )
        
        # Gather experimental evidence
        experimental_evidence = mutation.validation_method if mutation.experimentally_validated else None
        literature_support = []
        if mutation.metadata and 'pmid' in mutation.metadata:
            pmid = mutation.metadata['pmid']
            if pmid:
                literature_support.append(f"PMID:{pmid}")
        
        # Create analysis result
        result = MutationAnalysisResult(
            mutation=mutation,
            base_donor_score=base_donor_score,
            base_acceptor_score=base_acceptor_score,
            base_combined_score=base_combined_score,
            meta_donor_score=meta_donor_score,
            meta_acceptor_score=meta_acceptor_score,
            meta_combined_score=meta_combined_score,
            meta_prediction_class=meta_prediction_class,
            score_improvement=score_improvement,
            prediction_agreement=prediction_agreement,
            confidence_delta=confidence_delta,
            local_sequence=wt_sequence,
            cryptic_sites_detected=cryptic_sites,
            splice_strength_analysis=strength_analysis,
            experimental_evidence=experimental_evidence,
            literature_support=literature_support
        )
        
        # Save individual analysis
        self._save_mutation_analysis(result)
        
        self.logger.info(f"Analysis completed for {mutation.source_id}")
        return result
    
    def analyze_mutation_cohort(self, mutations: List[SpliceMutation]) -> Dict[str, Any]:
        """
        Analyze a cohort of related mutations.
        
        Parameters
        ----------
        mutations : List[SpliceMutation]
            List of mutations to analyze together
            
        Returns
        -------
        Dict[str, Any]
            Cohort analysis results
        """
        self.logger.info(f"Starting cohort analysis of {len(mutations)} mutations")
        
        # Analyze each mutation individually
        individual_results = []
        for mutation in mutations:
            try:
                result = self.analyze_single_mutation(mutation)
                individual_results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to analyze {mutation.source_id}: {e}")
                continue
        
        if not individual_results:
            self.logger.warning("No mutations were successfully analyzed")
            return {}
        
        # Cohort-level analysis
        cohort_analysis = {
            "total_mutations": len(mutations),
            "successfully_analyzed": len(individual_results),
            "analysis_success_rate": len(individual_results) / len(mutations),
            
            # Prediction performance
            "mean_base_score": np.mean([r.base_combined_score for r in individual_results]),
            "mean_meta_score": np.mean([r.meta_combined_score for r in individual_results]),
            "mean_score_improvement": np.mean([r.score_improvement for r in individual_results]),
            "prediction_agreement_rate": np.mean([r.prediction_agreement for r in individual_results]),
            
            # Clinical significance breakdown
            "pathogenic_count": sum(1 for r in individual_results 
                                  if r.mutation.clinical_significance == ClinicalSignificance.PATHOGENIC),
            "likely_pathogenic_count": sum(1 for r in individual_results 
                                         if r.mutation.clinical_significance == ClinicalSignificance.LIKELY_PATHOGENIC),
            "benign_count": sum(1 for r in individual_results 
                              if r.mutation.clinical_significance == ClinicalSignificance.BENIGN),
            
            # Splice event type breakdown
            "event_type_counts": {},
            
            # Cryptic site analysis
            "total_cryptic_sites": sum(len(r.cryptic_sites_detected) for r in individual_results),
            "novel_cryptic_sites": sum(sum(1 for site in r.cryptic_sites_detected if site["is_novel"]) 
                                     for r in individual_results),
            
            # Individual results
            "individual_results": individual_results
        }
        
        # Count event types
        for result in individual_results:
            event_type = result.mutation.splice_event_type.value
            cohort_analysis["event_type_counts"][event_type] = (
                cohort_analysis["event_type_counts"].get(event_type, 0) + 1
            )
        
        # Save cohort analysis
        self._save_cohort_analysis(cohort_analysis)
        
        self.logger.info(f"Cohort analysis completed. Success rate: {cohort_analysis['analysis_success_rate']:.2%}")
        return cohort_analysis
    
    def _save_mutation_analysis(self, result: MutationAnalysisResult) -> None:
        """Save individual mutation analysis results."""
        output_dir = self.work_dir / "individual_analyses"
        output_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        analysis_data = {
            "mutation_info": {
                "source_id": result.mutation.source_id,
                "gene_symbol": result.mutation.gene_symbol,
                "position": result.mutation.position,
                "change": f"{result.mutation.ref_allele}>{result.mutation.alt_allele}",
                "hgvs": result.mutation.hgvs_notation,
                "clinical_significance": result.mutation.clinical_significance.value if result.mutation.clinical_significance else None,
                "disease_context": result.mutation.disease_context
            },
            "prediction_scores": {
                "base_donor": result.base_donor_score,
                "base_acceptor": result.base_acceptor_score,
                "base_combined": result.base_combined_score,
                "meta_donor": result.meta_donor_score,
                "meta_acceptor": result.meta_acceptor_score,
                "meta_combined": result.meta_combined_score,
                "meta_class": result.meta_prediction_class
            },
            "analysis_metrics": {
                "score_improvement": result.score_improvement,
                "prediction_agreement": result.prediction_agreement,
                "confidence_delta": result.confidence_delta
            },
            "cryptic_sites": result.cryptic_sites_detected,
            "splice_strength": result.splice_strength_analysis,
            "experimental_evidence": result.experimental_evidence,
            "literature_support": result.literature_support
        }
        
        filename = f"{result.mutation.source_id}_analysis.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(analysis_data, f, indent=2)
    
    def _save_cohort_analysis(self, cohort_analysis: Dict[str, Any]) -> None:
        """Save cohort analysis results."""
        # Remove individual results for summary
        summary = cohort_analysis.copy()
        individual_results = summary.pop("individual_results", [])
        
        # Save summary
        with open(self.work_dir / "cohort_analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results as CSV
        if individual_results:
            detailed_data = []
            for result in individual_results:
                row = {
                    "mutation_id": result.mutation.source_id,
                    "gene_symbol": result.mutation.gene_symbol,
                    "position": result.mutation.position,
                    "change": f"{result.mutation.ref_allele}>{result.mutation.alt_allele}",
                    "clinical_significance": result.mutation.clinical_significance.value if result.mutation.clinical_significance else None,
                    "base_combined_score": result.base_combined_score,
                    "meta_combined_score": result.meta_combined_score,
                    "score_improvement": result.score_improvement,
                    "prediction_agreement": result.prediction_agreement,
                    "confidence_delta": result.confidence_delta,
                    "cryptic_sites_count": len(result.cryptic_sites_detected),
                    "experimental_evidence": result.experimental_evidence,
                    "database": result.mutation.source_database
                }
                detailed_data.append(row)
            
            df = pd.DataFrame(detailed_data)
            df.to_csv(self.work_dir / "cohort_detailed_results.csv", index=False)


# Convenience functions for common use cases
def analyze_cftr_cryptic_exon(work_dir: Path, meta_model_path: Optional[Path] = None) -> MutationAnalysisResult:
    """Analyze the famous CFTR c.3718-2477C>T cryptic exon mutation."""
    from ..data_sources.base import SpliceMutation, SpliceEventType, ClinicalSignificance
    
    # Create CFTR mutation object
    cftr_mutation = SpliceMutation(
        chrom="7",
        position=117199644,
        ref_allele="C",
        alt_allele="T",
        gene_id="ENSG00000001626",
        gene_symbol="CFTR",
        transcript_id="ENST00000003084",
        splice_event_type=SpliceEventType.PSEUDOEXON_ACTIVATION,
        affected_site_type="cryptic_donor",
        splice_site_position=117197167,
        clinical_significance=ClinicalSignificance.PATHOGENIC,
        disease_context="Cystic Fibrosis",
        experimentally_validated=True,
        validation_method="RT-PCR",
        source_database="case_study",
        source_id="CFTR_c.3718-2477C>T",
        hgvs_notation="c.3718-2477C>T"
    )
    
    workflow = MutationAnalysisWorkflow(work_dir, meta_model_path)
    return workflow.analyze_single_mutation(cftr_mutation)


def analyze_met_exon14_cohort(work_dir: Path, meta_model_path: Optional[Path] = None) -> Dict[str, Any]:
    """Analyze a cohort of MET exon 14 skipping mutations."""
    from ..data_sources.base import SpliceMutation, SpliceEventType, ClinicalSignificance
    
    # Create representative MET mutations
    met_mutations = [
        SpliceMutation(
            chrom="7", position=116412043, ref_allele="G", alt_allele="A",
            gene_id="ENSG00000105976", gene_symbol="MET",
            splice_event_type=SpliceEventType.EXON_SKIPPING,
            affected_site_type="donor", clinical_significance=ClinicalSignificance.PATHOGENIC,
            disease_context="Lung Cancer", experimentally_validated=True,
            source_database="case_study", source_id="MET_c.3028+1G>A",
            hgvs_notation="c.3028+1G>A"
        ),
        # Add more MET mutations as needed
    ]
    
    workflow = MutationAnalysisWorkflow(work_dir, meta_model_path)
    return workflow.analyze_mutation_cohort(met_mutations) 