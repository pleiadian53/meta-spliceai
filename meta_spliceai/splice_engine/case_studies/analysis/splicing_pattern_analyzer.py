"""
Alternative Splicing Pattern Analyzer

This module implements comprehensive algorithms for detecting and classifying
alternative splicing patterns from delta scores and splice site annotations.
It converts individual site-level predictions into coordinated splicing patterns
such as exon skipping, intron retention, and cryptic site usage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import networkx as nx


class SplicingPatternType(Enum):
    """Types of alternative splicing patterns."""
    EXON_SKIPPING = "exon_skipping"
    EXON_INCLUSION = "exon_inclusion"
    INTRON_RETENTION = "intron_retention"
    ALTERNATIVE_5SS = "alternative_5ss"  # Alternative 5' splice site
    ALTERNATIVE_3SS = "alternative_3ss"  # Alternative 3' splice site
    CRYPTIC_ACTIVATION = "cryptic_activation"
    CRYPTIC_SUPPRESSION = "cryptic_suppression"
    MUTUALLY_EXCLUSIVE = "mutually_exclusive_exons"
    ALTERNATIVE_FIRST = "alternative_first_exon"
    ALTERNATIVE_LAST = "alternative_last_exon"
    COMPLEX = "complex_pattern"
    CANONICAL_DISRUPTION = "canonical_disruption"


@dataclass
class SpliceSite:
    """Represents a splice site with delta scores."""
    position: int
    site_type: str  # 'donor' or 'acceptor'
    delta_score: float
    is_canonical: bool
    is_cryptic: bool
    strand: str
    gene_id: str
    transcript_id: Optional[str] = None
    exon_number: Optional[int] = None
    
    @property
    def is_activated(self) -> bool:
        """Check if site is activated (positive delta score)."""
        return self.delta_score > 0
    
    @property
    def is_suppressed(self) -> bool:
        """Check if site is suppressed (negative delta score)."""
        return self.delta_score < 0


@dataclass 
class SplicingPattern:
    """Represents a detected alternative splicing pattern."""
    pattern_type: SplicingPatternType
    affected_sites: List[SpliceSite]
    confidence_score: float
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    coordinates: Dict[str, int] = field(default_factory=dict)
    predicted_consequence: str = ""
    
    @property
    def pattern_id(self) -> str:
        """Generate unique pattern identifier."""
        sites_str = "_".join([f"{s.position}:{s.site_type[0]}" for s in self.affected_sites])
        return f"{self.pattern_type.value}_{sites_str}"
    
    @property
    def severity_score(self) -> float:
        """Calculate pattern severity based on delta scores."""
        if not self.affected_sites:
            return 0.0
        max_delta = max(abs(s.delta_score) for s in self.affected_sites)
        mean_delta = np.mean([abs(s.delta_score) for s in self.affected_sites])
        return (max_delta + mean_delta) / 2


class SplicingPatternAnalyzer:
    """
    Analyzes splice site delta scores to detect coordinated alternative
    splicing patterns.
    """
    
    # Thresholds for pattern detection
    DELTA_SCORE_THRESHOLD = 0.2
    PROXIMITY_WINDOW = 300  # bp window for nearby sites
    MIN_EXON_SIZE = 50
    MAX_EXON_SIZE = 5000
    
    def __init__(self, gene_annotations: Optional[pd.DataFrame] = None):
        """
        Initialize the pattern analyzer.
        
        Parameters
        ----------
        gene_annotations : pd.DataFrame, optional
            Gene/exon annotations for context
        """
        self.gene_annotations = gene_annotations
        self.detected_patterns = []
    
    def analyze_variant_impact(
        self,
        splice_sites: List[SpliceSite],
        variant_position: int,
        variant_type: str = "SNV"
    ) -> List[SplicingPattern]:
        """
        Analyze the impact of a variant on splicing patterns.
        
        Parameters
        ----------
        splice_sites : List[SpliceSite]
            List of splice sites with delta scores
        variant_position : int
            Position of the variant
        variant_type : str
            Type of variant (SNV, insertion, deletion)
            
        Returns
        -------
        List[SplicingPattern]
            Detected alternative splicing patterns
        """
        if not splice_sites:
            return []
        
        # Group sites by gene
        sites_by_gene = self._group_sites_by_gene(splice_sites)
        
        patterns = []
        for gene_id, gene_sites in sites_by_gene.items():
            # Sort sites by position
            gene_sites.sort(key=lambda x: x.position)
            
            # Detect different pattern types
            patterns.extend(self._detect_exon_skipping(gene_sites, variant_position))
            patterns.extend(self._detect_intron_retention(gene_sites, variant_position))
            patterns.extend(self._detect_alternative_splice_sites(gene_sites, variant_position))
            patterns.extend(self._detect_cryptic_patterns(gene_sites, variant_position))
            patterns.extend(self._detect_complex_patterns(gene_sites, variant_position))
        
        # Rank patterns by confidence and severity
        patterns = self._rank_patterns(patterns)
        
        # Store for later retrieval
        self.detected_patterns = patterns
        
        return patterns
    
    def _group_sites_by_gene(self, sites: List[SpliceSite]) -> Dict[str, List[SpliceSite]]:
        """Group splice sites by gene."""
        sites_by_gene = defaultdict(list)
        for site in sites:
            sites_by_gene[site.gene_id].append(site)
        return dict(sites_by_gene)
    
    def _detect_exon_skipping(
        self,
        sites: List[SpliceSite],
        variant_pos: int
    ) -> List[SplicingPattern]:
        """
        Detect exon skipping patterns.
        
        Looks for:
        - Suppressed donor and acceptor pairs (canonical sites)
        - Enhanced flanking sites
        """
        patterns = []
        
        # Find pairs of suppressed canonical sites
        for i in range(len(sites) - 1):
            if not sites[i].is_canonical or not sites[i+1].is_canonical:
                continue
                
            # Check for donor-acceptor pair
            if (sites[i].site_type == 'acceptor' and 
                sites[i+1].site_type == 'donor' and
                sites[i].is_suppressed and 
                sites[i+1].is_suppressed):
                
                # Calculate exon size
                exon_size = sites[i+1].position - sites[i].position
                
                if self.MIN_EXON_SIZE <= exon_size <= self.MAX_EXON_SIZE:
                    # Look for enhanced flanking sites
                    upstream_donor = self._find_nearest_site(
                        sites, sites[i].position, 'donor', direction='upstream'
                    )
                    downstream_acceptor = self._find_nearest_site(
                        sites, sites[i+1].position, 'acceptor', direction='downstream'
                    )
                    
                    affected_sites = [sites[i], sites[i+1]]
                    if upstream_donor and upstream_donor.is_activated:
                        affected_sites.append(upstream_donor)
                    if downstream_acceptor and downstream_acceptor.is_activated:
                        affected_sites.append(downstream_acceptor)
                    
                    # Calculate confidence
                    confidence = self._calculate_pattern_confidence(
                        affected_sites, 'exon_skipping'
                    )
                    
                    pattern = SplicingPattern(
                        pattern_type=SplicingPatternType.EXON_SKIPPING,
                        affected_sites=affected_sites,
                        confidence_score=confidence,
                        coordinates={
                            'exon_start': sites[i].position,
                            'exon_end': sites[i+1].position,
                            'exon_size': exon_size
                        },
                        predicted_consequence=f"Skipping of {exon_size}bp exon"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_intron_retention(
        self,
        sites: List[SpliceSite],
        variant_pos: int
    ) -> List[SplicingPattern]:
        """
        Detect intron retention patterns.
        
        Looks for suppressed donor-acceptor pairs without alternative sites.
        """
        patterns = []
        
        for i in range(len(sites) - 1):
            # Look for consecutive donor-acceptor pairs
            if (sites[i].site_type == 'donor' and 
                sites[i+1].site_type == 'acceptor' and
                sites[i].is_canonical and
                sites[i+1].is_canonical):
                
                # Both sites should be suppressed
                if sites[i].is_suppressed and sites[i+1].is_suppressed:
                    intron_size = sites[i+1].position - sites[i].position
                    
                    # Check for lack of alternative sites in between
                    intervening_sites = [
                        s for s in sites 
                        if sites[i].position < s.position < sites[i+1].position
                        and s.is_activated
                    ]
                    
                    if not intervening_sites:
                        confidence = self._calculate_pattern_confidence(
                            [sites[i], sites[i+1]], 'intron_retention'
                        )
                        
                        pattern = SplicingPattern(
                            pattern_type=SplicingPatternType.INTRON_RETENTION,
                            affected_sites=[sites[i], sites[i+1]],
                            confidence_score=confidence,
                            coordinates={
                                'intron_start': sites[i].position,
                                'intron_end': sites[i+1].position,
                                'intron_size': intron_size
                            },
                            predicted_consequence=f"Retention of {intron_size}bp intron"
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_alternative_splice_sites(
        self,
        sites: List[SpliceSite],
        variant_pos: int
    ) -> List[SplicingPattern]:
        """
        Detect alternative 5' or 3' splice site usage.
        
        Looks for activated alternative sites near suppressed canonical sites.
        """
        patterns = []
        
        for site in sites:
            if not site.is_canonical or not site.is_suppressed:
                continue
            
            # Find nearby alternative sites of same type
            nearby_sites = [
                s for s in sites
                if s.site_type == site.site_type
                and s != site
                and abs(s.position - site.position) <= self.PROXIMITY_WINDOW
                and s.is_activated
            ]
            
            for alt_site in nearby_sites:
                # Determine if 5' or 3' alternative
                if site.site_type == 'donor':
                    pattern_type = SplicingPatternType.ALTERNATIVE_5SS
                    shift = alt_site.position - site.position
                    consequence = f"Alternative 5'SS {shift:+d}bp from canonical"
                else:
                    pattern_type = SplicingPatternType.ALTERNATIVE_3SS
                    shift = alt_site.position - site.position
                    consequence = f"Alternative 3'SS {shift:+d}bp from canonical"
                
                confidence = self._calculate_pattern_confidence(
                    [site, alt_site], pattern_type.value
                )
                
                pattern = SplicingPattern(
                    pattern_type=pattern_type,
                    affected_sites=[site, alt_site],
                    confidence_score=confidence,
                    coordinates={
                        'canonical_position': site.position,
                        'alternative_position': alt_site.position,
                        'shift_distance': abs(shift)
                    },
                    predicted_consequence=consequence
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_cryptic_patterns(
        self,
        sites: List[SpliceSite],
        variant_pos: int
    ) -> List[SplicingPattern]:
        """
        Detect cryptic site activation or suppression patterns.
        """
        patterns = []
        
        # Find all cryptic sites
        cryptic_sites = [s for s in sites if s.is_cryptic]
        
        for site in cryptic_sites:
            if abs(site.delta_score) < self.DELTA_SCORE_THRESHOLD:
                continue
            
            if site.is_activated:
                pattern_type = SplicingPatternType.CRYPTIC_ACTIVATION
                consequence = f"Cryptic {site.site_type} activation at position {site.position}"
            else:
                pattern_type = SplicingPatternType.CRYPTIC_SUPPRESSION
                consequence = f"Cryptic {site.site_type} suppression at position {site.position}"
            
            # Find affected canonical sites
            nearest_canonical = self._find_nearest_canonical_site(
                sites, site.position, site.site_type
            )
            
            affected_sites = [site]
            if nearest_canonical:
                affected_sites.append(nearest_canonical)
            
            confidence = self._calculate_pattern_confidence(
                affected_sites, pattern_type.value
            )
            
            pattern = SplicingPattern(
                pattern_type=pattern_type,
                affected_sites=affected_sites,
                confidence_score=confidence,
                coordinates={
                    'cryptic_position': site.position,
                    'distance_from_variant': abs(site.position - variant_pos)
                },
                predicted_consequence=consequence
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_complex_patterns(
        self,
        sites: List[SpliceSite],
        variant_pos: int
    ) -> List[SplicingPattern]:
        """
        Detect complex splicing patterns involving multiple coordinated changes.
        """
        patterns = []
        
        # Build splice site graph
        graph = self._build_splice_graph(sites)
        
        # Find connected components with multiple affected sites
        components = list(nx.connected_components(graph))
        
        for component in components:
            if len(component) < 3:  # Need at least 3 sites for complex pattern
                continue
            
            component_sites = [s for s in sites if s.position in component]
            
            # Check for coordinated changes
            activated = [s for s in component_sites if s.is_activated]
            suppressed = [s for s in component_sites if s.is_suppressed]
            
            if len(activated) >= 2 and len(suppressed) >= 2:
                # Determine specific complex pattern type
                pattern_subtype = self._classify_complex_pattern(component_sites)
                
                confidence = self._calculate_pattern_confidence(
                    component_sites, 'complex'
                )
                
                pattern = SplicingPattern(
                    pattern_type=SplicingPatternType.COMPLEX,
                    affected_sites=component_sites,
                    confidence_score=confidence,
                    supporting_evidence={
                        'subtype': pattern_subtype,
                        'activated_sites': len(activated),
                        'suppressed_sites': len(suppressed)
                    },
                    predicted_consequence=f"Complex pattern: {pattern_subtype}"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _build_splice_graph(self, sites: List[SpliceSite]) -> nx.Graph:
        """Build a graph connecting related splice sites."""
        G = nx.Graph()
        
        # Add nodes
        for site in sites:
            G.add_node(site.position, site=site)
        
        # Add edges between potentially related sites
        for i, site1 in enumerate(sites):
            for site2 in sites[i+1:]:
                # Connect if within proximity and show coordinated changes
                if abs(site1.position - site2.position) <= self.PROXIMITY_WINDOW * 2:
                    if (site1.is_activated and site2.is_activated) or \
                       (site1.is_suppressed and site2.is_suppressed):
                        G.add_edge(site1.position, site2.position)
        
        return G
    
    def _classify_complex_pattern(self, sites: List[SpliceSite]) -> str:
        """Classify the specific type of complex pattern."""
        # Analyze site arrangement
        donors = [s for s in sites if s.site_type == 'donor']
        acceptors = [s for s in sites if s.site_type == 'acceptor']
        
        if len(donors) > 2 and len(acceptors) > 2:
            return "multiple_exon_changes"
        elif any(s.is_cryptic for s in sites) and any(s.is_canonical for s in sites):
            return "mixed_canonical_cryptic"
        else:
            return "unclassified_complex"
    
    def _find_nearest_site(
        self,
        sites: List[SpliceSite],
        position: int,
        site_type: str,
        direction: str = 'both'
    ) -> Optional[SpliceSite]:
        """Find the nearest splice site of given type."""
        candidates = [s for s in sites if s.site_type == site_type]
        
        if direction == 'upstream':
            candidates = [s for s in candidates if s.position < position]
        elif direction == 'downstream':
            candidates = [s for s in candidates if s.position > position]
        
        if not candidates:
            return None
        
        return min(candidates, key=lambda s: abs(s.position - position))
    
    def _find_nearest_canonical_site(
        self,
        sites: List[SpliceSite],
        position: int,
        site_type: str
    ) -> Optional[SpliceSite]:
        """Find the nearest canonical splice site."""
        candidates = [
            s for s in sites 
            if s.site_type == site_type and s.is_canonical
        ]
        
        if not candidates:
            return None
        
        return min(candidates, key=lambda s: abs(s.position - position))
    
    def _calculate_pattern_confidence(
        self,
        sites: List[SpliceSite],
        pattern_type: str
    ) -> float:
        """
        Calculate confidence score for a detected pattern.
        
        Considers:
        - Magnitude of delta scores
        - Number of supporting sites
        - Pattern-specific features
        """
        if not sites:
            return 0.0
        
        # Base confidence from delta scores
        max_delta = max(abs(s.delta_score) for s in sites)
        mean_delta = np.mean([abs(s.delta_score) for s in sites])
        
        base_confidence = min(1.0, (max_delta + mean_delta) / 2)
        
        # Adjust for pattern-specific features
        if pattern_type == 'exon_skipping':
            # Higher confidence if both splice sites strongly affected
            if len(sites) >= 2 and all(abs(s.delta_score) > 0.5 for s in sites[:2]):
                base_confidence *= 1.2
        elif pattern_type == 'cryptic_activation':
            # Higher confidence for strong cryptic activation
            if any(s.delta_score > 0.8 for s in sites if s.is_cryptic):
                base_confidence *= 1.15
        
        return min(1.0, base_confidence)
    
    def _rank_patterns(self, patterns: List[SplicingPattern]) -> List[SplicingPattern]:
        """Rank patterns by confidence and biological significance."""
        # Define pattern priority scores
        priority_scores = {
            SplicingPatternType.EXON_SKIPPING: 1.0,
            SplicingPatternType.INTRON_RETENTION: 0.9,
            SplicingPatternType.CRYPTIC_ACTIVATION: 0.85,
            SplicingPatternType.ALTERNATIVE_5SS: 0.8,
            SplicingPatternType.ALTERNATIVE_3SS: 0.8,
            SplicingPatternType.COMPLEX: 0.7,
            SplicingPatternType.CANONICAL_DISRUPTION: 0.95,
        }
        
        for pattern in patterns:
            priority = priority_scores.get(pattern.pattern_type, 0.5)
            pattern.supporting_evidence['rank_score'] = (
                pattern.confidence_score * 0.7 + 
                priority * 0.3
            )
        
        # Sort by rank score
        patterns.sort(
            key=lambda p: p.supporting_evidence.get('rank_score', 0),
            reverse=True
        )
        
        return patterns
    
    def summarize_patterns(self, patterns: List[SplicingPattern]) -> pd.DataFrame:
        """
        Create a summary DataFrame of detected patterns.
        
        Parameters
        ----------
        patterns : List[SplicingPattern]
            List of detected patterns
            
        Returns
        -------
        pd.DataFrame
            Summary of patterns with key metrics
        """
        if not patterns:
            return pd.DataFrame()
        
        summary_data = []
        for pattern in patterns:
            summary_data.append({
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type.value,
                'confidence': pattern.confidence_score,
                'severity': pattern.severity_score,
                'num_sites': len(pattern.affected_sites),
                'consequence': pattern.predicted_consequence,
                'rank_score': pattern.supporting_evidence.get('rank_score', 0)
            })
        
        return pd.DataFrame(summary_data)


def demonstrate_pattern_analysis():
    """Demonstrate alternative splicing pattern analysis."""
    print("\n" + "="*60)
    print("Alternative Splicing Pattern Analysis Demonstration")
    print("="*60 + "\n")
    
    # Create example splice sites with delta scores
    example_sites = [
        SpliceSite(100, 'donor', -0.8, True, False, '+', 'GENE1', exon_number=1),
        SpliceSite(200, 'acceptor', -0.7, True, False, '+', 'GENE1', exon_number=2),
        SpliceSite(150, 'donor', 0.6, False, True, '+', 'GENE1'),  # Cryptic
        SpliceSite(300, 'donor', 0.3, True, False, '+', 'GENE1', exon_number=2),
        SpliceSite(400, 'acceptor', 0.2, True, False, '+', 'GENE1', exon_number=3),
    ]
    
    # Analyze patterns
    analyzer = SplicingPatternAnalyzer()
    patterns = analyzer.analyze_variant_impact(
        example_sites,
        variant_position=125,
        variant_type='SNV'
    )
    
    print(f"📊 Detected {len(patterns)} splicing patterns:\n")
    
    for i, pattern in enumerate(patterns, 1):
        print(f"{i}. {pattern.pattern_type.value.upper()}")
        print(f"   Confidence: {pattern.confidence_score:.2f}")
        print(f"   Severity: {pattern.severity_score:.2f}")
        print(f"   Affected sites: {len(pattern.affected_sites)}")
        print(f"   Consequence: {pattern.predicted_consequence}")
        print()
    
    # Generate summary
    summary_df = analyzer.summarize_patterns(patterns)
    if not summary_df.empty:
        print("\n📈 Pattern Summary:")
        print(summary_df.to_string(index=False))
    
    print("\n✅ Pattern analysis demonstration complete!")


if __name__ == "__main__":
    demonstrate_pattern_analysis()
