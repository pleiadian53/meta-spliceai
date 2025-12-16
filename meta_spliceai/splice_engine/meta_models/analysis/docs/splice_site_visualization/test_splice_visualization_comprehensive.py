#!/usr/bin/env python3
"""
Comprehensive test script for splice site visualization system.

This script tests all major functionality:
1. Data loading and validation
2. Single gene visualization
3. Multi-gene comparison
4. Meta-learning improvement detection
5. Report generation
"""

import sys
from pathlib import Path
sys.path.append('.')

from meta_spliceai.splice_engine.meta_models.analysis.splice_site_comparison_visualizer import SpliceSiteComparisonVisualizer

def test_data_loading():
    """Test data loading functionality."""
    print("🧪 Testing Data Loading")
    print("=" * 50)
    
    viz = SpliceSiteComparisonVisualizer(verbose=True)
    
    # Test 1: Load gene features
    print("\n1. Loading gene features...")
    gene_features = viz.load_gene_features('data/ensembl/spliceai_analysis/gene_features.tsv')
    assert gene_features is not None, "Failed to load gene features"
    print(f"   ✅ Loaded {len(gene_features)} gene features")
    
    # Test 2: Load training data with hierarchical sampling
    print("\n2. Loading training data...")
    base_data = viz.load_dataset('train_pc_1000/master', sample_genes=50)
    assert base_data is not None, "Failed to load base data"
    assert len(base_data) > 0, "Base data is empty"
    print(f"   ✅ Loaded {len(base_data):,} positions from {base_data['gene_id'].nunique()} genes")
    
    # Test 3: Load CV results
    print("\n3. Loading CV results...")
    cv_results = viz.load_cv_results('results/gene_cv_1000_run_15/position_level_classification_results.tsv')
    assert cv_results is not None, "Failed to load CV results"
    print(f"   ✅ Loaded {len(cv_results):,} CV results")
    
    # Test 4: Format meta data
    print("\n4. Formatting meta data...")
    meta_data = viz.format_cv_results_as_meta_data(cv_results)
    assert meta_data is not None, "Failed to format meta data"
    print(f"   ✅ Formatted {len(meta_data):,} meta predictions")
    
    return viz, base_data, meta_data

def test_gene_analysis(viz, base_data, meta_data):
    """Test gene-level analysis functionality."""
    print("\n🧪 Testing Gene Analysis")
    print("=" * 50)
    
    # Find genes with good splice site representation
    print("\n1. Finding suitable genes...")
    training_genes = set(base_data['gene_id'].unique())
    meta_genes = set(meta_data['gene_id'].unique())
    common_genes = training_genes & meta_genes
    
    good_genes = []
    for gene_id in list(common_genes)[:20]:  # Test first 20 genes
        gene_subset = base_data[base_data['gene_id'] == gene_id]
        donor_count = (gene_subset['splice_type'] == 'donor').sum()
        acceptor_count = (gene_subset['splice_type'] == 'acceptor').sum()
        
        if donor_count >= 3 and acceptor_count >= 3:
            gene_name = viz.get_gene_display_name(gene_id)
            good_genes.append({
                'gene_id': gene_id,
                'gene_name': gene_name,
                'donors': donor_count,
                'acceptors': acceptor_count
            })
    
    assert len(good_genes) > 0, "No suitable genes found"
    print(f"   ✅ Found {len(good_genes)} suitable genes")
    
    # Test gene data extraction
    print("\n2. Testing gene data extraction...")
    test_gene = good_genes[0]
    gene_data_base = viz.get_gene_data(base_data, test_gene['gene_id'])
    gene_data_meta = viz.get_gene_data(meta_data, test_gene['gene_id'])
    
    assert len(gene_data_base) > 0, "Failed to extract base gene data"
    assert len(gene_data_meta) > 0, "Failed to extract meta gene data"
    print(f"   ✅ Extracted data for {test_gene['gene_name']}: {len(gene_data_base)} base, {len(gene_data_meta)} meta positions")
    
    # Test change detection
    print("\n3. Testing change detection...")
    changes = viz.identify_splice_site_changes(gene_data_base, gene_data_meta, threshold=0.5)
    
    total_changes = (len(changes['rescued_donors']) + len(changes['rescued_acceptors']) + 
                    len(changes['eliminated_fp_donors']) + len(changes['eliminated_fp_acceptors']))
    
    print(f"   ✅ Detected {total_changes} total changes for {test_gene['gene_name']}")
    print(f"      • Rescued donors: {len(changes['rescued_donors'])}")
    print(f"      • Rescued acceptors: {len(changes['rescued_acceptors'])}")
    print(f"      • Eliminated FP donors: {len(changes['eliminated_fp_donors'])}")
    print(f"      • Eliminated FP acceptors: {len(changes['eliminated_fp_acceptors'])}")
    
    return good_genes

def test_visualization_generation(viz, base_data, meta_data, good_genes):
    """Test visualization generation."""
    print("\n🧪 Testing Visualization Generation")
    print("=" * 50)
    
    output_dir = Path('results/test_splice_visualization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test 1: Single gene visualization
    print("\n1. Creating single gene visualization...")
    test_gene = good_genes[0]
    gene_data_base = viz.get_gene_data(base_data, test_gene['gene_id'])
    gene_data_meta = viz.get_gene_data(meta_data, test_gene['gene_id'])
    
    result = viz.create_gene_comparison_plot(
        gene_data_base, gene_data_meta, 
        test_gene['gene_id'], output_dir, 
        threshold=0.5
    )
    
    plot_file = Path(result['plot_file'])
    assert plot_file.exists(), "Single gene plot not created"
    print(f"   ✅ Created single gene plot: {plot_file.name}")
    
    # Test 2: Multi-gene visualization (if we have enough genes)
    if len(good_genes) >= 3:
        print("\n2. Creating multi-gene visualization...")
        selected_genes = [g['gene_id'] for g in good_genes[:3]]
        
        result = viz.create_multi_gene_comparison(
            base_data, meta_data, 
            selected_genes, output_dir, 
            threshold=0.5
        )
        
        plot_file = Path(result['plot_file'])
        assert plot_file.exists(), "Multi-gene plot not created"
        print(f"   ✅ Created multi-gene plot: {plot_file.name}")
    
    # Test 3: Summary report
    print("\n3. Creating summary report...")
    all_changes = {}
    for gene in good_genes[:5]:  # Test first 5 genes
        try:
            gene_data_base = viz.get_gene_data(base_data, gene['gene_id'])
            gene_data_meta = viz.get_gene_data(meta_data, gene['gene_id'])
            changes = viz.identify_splice_site_changes(gene_data_base, gene_data_meta, threshold=0.5)
            all_changes[gene['gene_id']] = changes
        except ValueError:
            continue
    
    report_file = viz.generate_summary_report(all_changes, output_dir)
    assert Path(report_file).exists(), "Summary report not created"
    print(f"   ✅ Created summary report: {Path(report_file).name}")
    
    return output_dir

def test_edge_cases(viz, base_data, meta_data):
    """Test edge cases and error handling."""
    print("\n🧪 Testing Edge Cases")
    print("=" * 50)
    
    # Test 1: Non-existent gene
    print("\n1. Testing non-existent gene...")
    try:
        viz.get_gene_data(base_data, 'NONEXISTENT_GENE')
        assert False, "Should have raised ValueError"
    except ValueError:
        print("   ✅ Correctly handled non-existent gene")
    
    # Test 2: Empty meta data
    print("\n2. Testing with empty meta data...")
    changes = viz.identify_splice_site_changes(base_data.head(10), None, threshold=0.5)
    assert all(len(change_list) == 0 for change_list in changes.values()), "Should have no changes with empty meta data"
    print("   ✅ Correctly handled empty meta data")
    
    # Test 3: Different thresholds
    print("\n3. Testing different thresholds...")
    test_gene_id = base_data['gene_id'].iloc[0]
    gene_data_base = viz.get_gene_data(base_data, test_gene_id)
    
    try:
        gene_data_meta = viz.get_gene_data(meta_data, test_gene_id)
        changes_low = viz.identify_splice_site_changes(gene_data_base, gene_data_meta, threshold=0.1)
        changes_high = viz.identify_splice_site_changes(gene_data_base, gene_data_meta, threshold=0.9)
        print("   ✅ Successfully tested different thresholds")
    except ValueError:
        print("   ⚠️  Gene not found in meta data, skipping threshold test")

def main():
    """Run comprehensive tests."""
    print("🧬 Comprehensive Splice Site Visualization Test")
    print("=" * 60)
    
    try:
        # Test data loading
        viz, base_data, meta_data = test_data_loading()
        
        # Test gene analysis
        good_genes = test_gene_analysis(viz, base_data, meta_data)
        
        # Test visualization generation
        output_dir = test_visualization_generation(viz, base_data, meta_data, good_genes)
        
        # Test edge cases
        test_edge_cases(viz, base_data, meta_data)
        
        print("\n🎉 All Tests Passed!")
        print("=" * 60)
        print(f"📊 Test Results Summary:")
        print(f"   • Data loading: ✅ PASSED")
        print(f"   • Gene analysis: ✅ PASSED")
        print(f"   • Visualization generation: ✅ PASSED")
        print(f"   • Edge case handling: ✅ PASSED")
        print(f"   • Output directory: {output_dir}")
        
        # Show generated files
        print(f"\n📁 Generated Files:")
        for file in output_dir.glob("*"):
            if file.is_file():
                print(f"   • {file.name}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 