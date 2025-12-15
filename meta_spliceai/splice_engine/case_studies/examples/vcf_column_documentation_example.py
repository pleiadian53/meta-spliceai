#!/usr/bin/env python3
"""
VCF Column Documentation Example

Demonstrates how to use the VCF Column Documenter tool to analyze
ClinVar VCF files and generate comprehensive column documentation.

This example shows:
1. Basic usage with ClinVar VCF
2. Custom configuration options
3. Programmatic access to documentation
4. Integration with existing workflows
"""

import sys
from pathlib import Path

# Add the tools directory to the path
sys.path.append(str(Path(__file__).parent.parent / "tools"))

from vcf_column_documenter import VCFColumnDocumenter, VCFDocumentationConfig


def example_basic_usage():
    """Example: Basic usage with ClinVar VCF."""
    print("🔍 Example 1: Basic ClinVar VCF Documentation")
    print("=" * 50)
    
    # Configuration for basic analysis
    config = VCFDocumentationConfig(
        input_vcf=Path("data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz"),
        output_dir=Path("docs/clinvar_columns"),
        max_variants=10000,  # Limit for faster analysis
        sample_size=5000,
        output_formats=['json', 'markdown'],
        verbose=True
    )
    
    # Create documenter
    documenter = VCFColumnDocumenter(config)
    
    # Analyze columns
    documentation = documenter.analyze_vcf_columns()
    
    # Save documentation
    documenter.save_documentation()
    
    print(f"✅ Documented {len(documentation)} columns")
    print(f"📁 Output saved to: {config.output_dir}")


def example_programmatic_access():
    """Example: Programmatic access to documentation."""
    print("\n🔍 Example 2: Programmatic Access to Documentation")
    print("=" * 50)
    
    config = VCFDocumentationConfig(
        input_vcf=Path("data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz"),
        output_dir=Path("docs/programmatic"),
        max_variants=5000,
        verbose=False
    )
    
    documenter = VCFColumnDocumenter(config)
    documentation = documenter.analyze_vcf_columns()
    
    # Access specific column information
    print("\n📊 Column Analysis Results:")
    print("-" * 30)
    
    for col_name, col_doc in documentation.items():
        if col_name in ['CLNSIG', 'MC', 'TYPE']:  # Key columns
            print(f"\n{col_name}:")
            print(f"  Description: {col_doc.description}")
            print(f"  Data Type: {col_doc.data_type}")
            print(f"  Unique Values: {col_doc.unique_count}")
            print(f"  Top Values: {list(col_doc.value_counts.keys())[:5]}")
    
    # Generate structured output
    structured_output = documenter.generate_structured_output()
    print(f"\n📈 Total Columns Analyzed: {structured_output['metadata']['total_columns']}")
    print(f"📊 Sample Size: {structured_output['metadata']['sample_size']:,}")


def example_clinvar_specific_analysis():
    """Example: ClinVar-specific analysis focusing on key fields."""
    print("\n🔍 Example 3: ClinVar-Specific Analysis")
    print("=" * 50)
    
    config = VCFDocumentationConfig(
        input_vcf=Path("data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz"),
        output_dir=Path("docs/clinvar_specific"),
        max_variants=20000,
        sample_size=10000,
        output_formats=['json', 'markdown'],
        verbose=True
    )
    
    documenter = VCFColumnDocumenter(config)
    documentation = documenter.analyze_vcf_columns()
    
    # Focus on ClinVar-specific fields
    clinvar_fields = ['CLNSIG', 'CLNREVSTAT', 'MC', 'CLNDN', 'ORIGIN', 'SSR']
    
    print("\n🧬 ClinVar-Specific Field Analysis:")
    print("-" * 40)
    
    for field in clinvar_fields:
        if field in documentation:
            col_doc = documentation[field]
            print(f"\n{field} ({col_doc.description}):")
            print(f"  Total Values: {len(col_doc.value_counts):,}")
            print(f"  Unique Values: {col_doc.unique_count:,}")
            
            # Show top 5 values
            top_values = sorted(col_doc.value_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print("  Top Values:")
            for value, count in top_values:
                percentage = (count / sum(col_doc.value_counts.values())) * 100
                print(f"    {value}: {count:,} ({percentage:.1f}%)")
    
    # Save documentation
    documenter.save_documentation()


def example_integration_with_workflows():
    """Example: Integration with existing MetaSpliceAI workflows."""
    print("\n🔍 Example 4: Integration with MetaSpliceAI Workflows")
    print("=" * 50)
    
    # This shows how the documentation tool can be integrated
    # with existing VCF parsing workflows
    
    config = VCFDocumentationConfig(
        input_vcf=Path("data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz"),
        output_dir=Path("docs/workflow_integration"),
        max_variants=10000,
        verbose=True
    )
    
    documenter = VCFColumnDocumenter(config)
    documentation = documenter.analyze_vcf_columns()
    
    # Generate documentation that can be used by other tools
    structured_output = documenter.generate_structured_output()
    
    # Save metadata for other tools to use
    metadata_file = config.output_dir / "column_metadata.json"
    import json
    with open(metadata_file, 'w') as f:
        json.dump(structured_output, f, indent=2)
    
    print(f"✅ Generated metadata for workflow integration: {metadata_file}")
    
    # Example: Use the documentation to inform parsing decisions
    print("\n🔧 Workflow Integration Example:")
    print("-" * 35)
    
    # Check if key fields are present
    required_fields = ['CLNSIG', 'MC', 'TYPE']
    missing_fields = [field for field in required_fields if field not in documentation]
    
    if missing_fields:
        print(f"⚠️  Missing required fields: {missing_fields}")
    else:
        print("✅ All required fields present")
    
    # Check field quality
    for field in required_fields:
        if field in documentation:
            col_doc = documentation[field]
            null_percentage = (col_doc.null_count / (col_doc.null_count + sum(col_doc.value_counts.values()))) * 100
            print(f"  {field}: {null_percentage:.1f}% null values")


def main():
    """Run all examples."""
    print("🧬 VCF Column Documentation Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_usage()
        example_programmatic_access()
        example_clinvar_specific_analysis()
        example_integration_with_workflows()
        
        print("\n✅ All examples completed successfully!")
        print("\n📚 Generated Documentation:")
        print("  - docs/clinvar_columns/: Basic ClinVar analysis")
        print("  - docs/programmatic/: Programmatic access example")
        print("  - docs/clinvar_specific/: ClinVar-specific analysis")
        print("  - docs/workflow_integration/: Workflow integration example")
        
    except FileNotFoundError as e:
        print(f"❌ Error: VCF file not found: {e}")
        print("Please ensure the ClinVar VCF file exists at the specified path.")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


