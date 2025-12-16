#!/bin/bash
# Quick Splice Site Analysis Script
# 
# This script provides fast analysis of splice_sites.tsv using standard Unix tools
# Perfect for quick insights without requiring Python dependencies
#
# Documentation: docs/data/splice_sites/splice_site_annotations.md
# Output: scripts/data/output/splice_sites_quick_summary.txt
#
# Usage: ./scripts/data/quick_splice_analysis.sh [path_to_splice_sites.tsv]

# Default file path
SPLICE_FILE="${1:-/Users/pleiadian53/work/meta-spliceai/data/ensembl/splice_sites.tsv}"
OUTPUT_DIR="/Users/pleiadian53/work/meta-spliceai/scripts/data/output"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🧬 QUICK SPLICE SITE ANALYSIS"
echo "================================="
echo "Data file: $SPLICE_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Generated: $(date)"
echo ""

# Check if file exists
if [[ ! -f "$SPLICE_FILE" ]]; then
    echo "❌ Error: File not found: $SPLICE_FILE"
    exit 1
fi

echo "📊 BASIC STATISTICS"
echo "-------------------"
total_lines=$(wc -l < "$SPLICE_FILE")
total_sites=$((total_lines - 1))  # Subtract header
echo "Total splice sites: $(printf "%'d" $total_sites)"

unique_genes=$(tail -n +2 "$SPLICE_FILE" | cut -f7 | sort -u | wc -l)
echo "Unique genes: $(printf "%'d" $unique_genes)"

unique_transcripts=$(tail -n +2 "$SPLICE_FILE" | cut -f8 | sort -u | wc -l)
echo "Unique transcripts: $(printf "%'d" $unique_transcripts)"

unique_chromosomes=$(tail -n +2 "$SPLICE_FILE" | cut -f1 | sort -u | wc -l)
echo "Unique chromosomes: $unique_chromosomes"

# Calculate averages
avg_transcripts_per_gene=$(echo "scale=2; $unique_transcripts / $unique_genes" | bc -l)
avg_sites_per_gene=$(echo "scale=1; $total_sites / $unique_genes" | bc -l)
echo "Average transcripts per gene: $avg_transcripts_per_gene"
echo "Average splice sites per gene: $avg_sites_per_gene"

echo ""
echo "🧭 SPLICE SITE TYPES"
echo "--------------------"
tail -n +2 "$SPLICE_FILE" | cut -f6 | sort | uniq -c | sort -nr | while read count type; do
    percentage=$(echo "scale=1; ($count * 100) / $total_sites" | bc -l)
    printf "%-10s: %'8d sites (%.1f%%)\n" "$type" "$count" "$percentage"
done

echo ""
echo "🧬 TOP 15 CHROMOSOMES BY SPLICE SITE COUNT"
echo "-------------------------------------------"
tail -n +2 "$SPLICE_FILE" | cut -f1 | sort | uniq -c | sort -nr | head -15 | while read count chrom; do
    percentage=$(echo "scale=2; ($count * 100) / $total_sites" | bc -l)
    printf "Chr %-2s: %'8d sites (%.2f%%)\n" "$chrom" "$count" "$percentage"
done

echo ""
echo "🏆 TOP 20 GENES BY SPLICE SITE COUNT"
echo "------------------------------------"
echo "Rank | Gene ID           | Splice Sites"
echo "-----+-------------------+-------------"
tail -n +2 "$SPLICE_FILE" | cut -f7 | sort | uniq -c | sort -nr | head -20 | \
awk 'BEGIN{rank=1} {printf "%4d | %-17s | %11s\n", rank, $2, $1; rank++}'

echo ""
echo "📈 SPLICE SITES PER GENE DISTRIBUTION"
echo "--------------------------------------"
temp_file=$(mktemp)
tail -n +2 "$SPLICE_FILE" | cut -f7 | sort | uniq -c | awk '{print $1}' > "$temp_file"

total_genes=$(wc -l < "$temp_file")

echo "Distribution analysis:"
echo "1 site:      $(awk '$1==1' "$temp_file" | wc -l) genes ($(echo "scale=1; $(awk '$1==1' "$temp_file" | wc -l) * 100 / $total_genes" | bc -l)%)"
echo "2-5 sites:   $(awk '$1>=2 && $1<=5' "$temp_file" | wc -l) genes ($(echo "scale=1; $(awk '$1>=2 && $1<=5' "$temp_file" | wc -l) * 100 / $total_genes" | bc -l)%)"
echo "6-10 sites:  $(awk '$1>=6 && $1<=10' "$temp_file" | wc -l) genes ($(echo "scale=1; $(awk '$1>=6 && $1<=10' "$temp_file" | wc -l) * 100 / $total_genes" | bc -l)%)"
echo "11-25 sites: $(awk '$1>=11 && $1<=25' "$temp_file" | wc -l) genes ($(echo "scale=1; $(awk '$1>=11 && $1<=25' "$temp_file" | wc -l) * 100 / $total_genes" | bc -l)%)"
echo "26-50 sites: $(awk '$1>=26 && $1<=50' "$temp_file" | wc -l) genes ($(echo "scale=1; $(awk '$1>=26 && $1<=50' "$temp_file" | wc -l) * 100 / $total_genes" | bc -l)%)"
echo "51-100 sites: $(awk '$1>=51 && $1<=100' "$temp_file" | wc -l) genes ($(echo "scale=1; $(awk '$1>=51 && $1<=100' "$temp_file" | wc -l) * 100 / $total_genes" | bc -l)%)"
echo ">100 sites:  $(awk '$1>100' "$temp_file" | wc -l) genes ($(echo "scale=1; $(awk '$1>100' "$temp_file" | wc -l) * 100 / $total_genes" | bc -l)%)"

rm "$temp_file"

echo ""
echo "🧬 TOP 15 GENES BY TRANSCRIPT COUNT (ISOFORMS)"
echo "-----------------------------------------------"
echo "Rank | Gene ID           | Transcripts | Splice Sites"
echo "-----+-------------------+-------------+--------------"

# Create temporary file for transcript counts per gene
temp_transcripts=$(mktemp)
tail -n +2 "$SPLICE_FILE" | awk '{print $7 "\t" $8}' | sort -u | cut -f1 | sort | uniq -c > "$temp_transcripts"

# Get top genes by transcript count and add splice site counts
sort -nr "$temp_transcripts" | head -15 | while read transcript_count gene_id; do
    splice_count=$(tail -n +2 "$SPLICE_FILE" | awk -v gene="$gene_id" '$7==gene' | wc -l)
    printf "%4d | %-17s | %11s | %12s\n" "$((++rank))" "$gene_id" "$transcript_count" "$splice_count"
done

rm "$temp_transcripts"

echo ""
echo "📊 GENERATING SUMMARY REPORT"
echo "----------------------------"

# Generate summary report
summary_file="$OUTPUT_DIR/splice_sites_quick_summary.txt"
{
    echo "SPLICE SITES QUICK ANALYSIS SUMMARY"
    echo "==================================="
    echo "Generated: $(date)"
    echo "Data source: $SPLICE_FILE"
    echo ""
    echo "OVERVIEW STATISTICS"
    echo "------------------"
    echo "Total splice sites:      $(printf "%'d" $total_sites)"
    echo "Unique genes:           $(printf "%'d" $unique_genes)"
    echo "Unique transcripts:     $(printf "%'d" $unique_transcripts)"
    echo "Unique chromosomes:     $unique_chromosomes"
    echo "Avg transcripts/gene:   $avg_transcripts_per_gene"
    echo "Avg splice sites/gene:  $avg_sites_per_gene"
    echo ""
    echo "SPLICE SITE TYPE BREAKDOWN"
    echo "--------------------------"
    tail -n +2 "$SPLICE_FILE" | cut -f6 | sort | uniq -c | sort -nr | while read count type; do
        percentage=$(echo "scale=1; ($count * 100) / $total_sites" | bc -l)
        printf "%-10s: %'8d sites (%.1f%%)\n" "$type" "$count" "$percentage"
    done
    echo ""
    echo "TOP 10 GENES BY SPLICE SITE COUNT"
    echo "---------------------------------"
    tail -n +2 "$SPLICE_FILE" | cut -f7 | sort | uniq -c | sort -nr | head -10 | while read count gene; do
        printf "%-17s: %'8d sites\n" "$gene" "$count"
    done
} > "$summary_file"

echo "✅ Quick analysis complete!"
echo "📄 Summary report saved to: $summary_file"
echo ""
echo "🚀 For detailed analysis with visualizations, run:"
echo "   python scripts/data/analyze_splice_sites.py"