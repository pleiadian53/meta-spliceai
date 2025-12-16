#!/bin/bash
# Gene Pattern Analysis Script
# 
# This script focuses on gene-level biological insights from splice_sites.tsv
# Provides detailed analysis of alternative splicing patterns and gene complexity
#
# Documentation: docs/data/splice_sites/splice_site_annotations.md
# Output: scripts/data/output/gene_analysis_report.txt
#
# Usage: ./scripts/data/analyze_gene_patterns.sh [path_to_splice_sites.tsv]

SPLICE_FILE="${1:-/Users/pleiadian53/work/meta-spliceai/data/ensembl/splice_sites.tsv}"
OUTPUT_DIR="/Users/pleiadian53/work/meta-spliceai/scripts/data/output"
mkdir -p "$OUTPUT_DIR"

echo "🧬 GENE PATTERN ANALYSIS"
echo "========================"
echo "Analyzing: $SPLICE_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# Create detailed gene analysis report
gene_report="$OUTPUT_DIR/gene_analysis_report.txt"

{
    echo "SPLICE SITE GENE ANALYSIS REPORT"
    echo "================================"
    echo "Generated: $(date)"
    echo "Data source: $(basename $SPLICE_FILE)"
    echo ""
    
    echo "ALTERNATIVE SPLICING COMPLEXITY"
    echo "------------------------------"
    echo "Genes ranked by number of transcript isoforms:"
    echo ""
    
    # Count transcripts per gene and get top 25
    temp_gene_transcripts=$(mktemp)
    tail -n +2 "$SPLICE_FILE" | awk '{print $7 "\t" $8}' | sort -u | \
    awk '{count[$1]++} END {for (gene in count) print count[gene], gene}' | \
    sort -nr > "$temp_gene_transcripts"
    
    echo "Rank | Gene ID           | Transcripts | Splice Sites | Sites/Transcript"
    echo "-----+-------------------+-------------+--------------+-----------------"
    
    head -25 "$temp_gene_transcripts" | while read transcript_count gene_id; do
        splice_count=$(tail -n +2 "$SPLICE_FILE" | awk -v gene="$gene_id" '$7==gene' | wc -l)
        sites_per_transcript=$(echo "scale=1; $splice_count / $transcript_count" | bc -l)
        rank=$((rank + 1))
        printf "%4d | %-17s | %11s | %12s | %15s\n" "$rank" "$gene_id" "$transcript_count" "$splice_count" "$sites_per_transcript"
    done
    
    rm "$temp_gene_transcripts"
    
    echo ""
    echo "SPLICE SITE DENSITY ANALYSIS"
    echo "----------------------------"
    echo "Genes with highest splice site density (sites per transcript):"
    echo ""
    
    # Calculate splice site density
    temp_density=$(mktemp)
    tail -n +2 "$SPLICE_FILE" | awk '{genes[$7]++; transcripts[$7][$8]=1} 
    END {
        for (gene in genes) {
            t_count = 0; 
            for (t in transcripts[gene]) t_count++; 
            if (t_count > 0) 
                print genes[gene]/t_count, gene, genes[gene], t_count
        }
    }' | sort -nr > "$temp_density"
    
    echo "Rank | Gene ID           | Density | Splice Sites | Transcripts"
    echo "-----+-------------------+---------+--------------+------------"
    
    head -20 "$temp_density" | while read density gene_id splice_count transcript_count; do
        rank=$((rank + 1))
        printf "%4d | %-17s | %7.1f | %12s | %11s\n" "$rank" "$gene_id" "$density" "$splice_count" "$transcript_count"
    done
    
    rm "$temp_density"
    
    echo ""
    echo "GENE SIZE ANALYSIS (by splice sites)"
    echo "-----------------------------------"
    
    # Analyze gene size distribution
    temp_sizes=$(mktemp)
    tail -n +2 "$SPLICE_FILE" | cut -f7 | sort | uniq -c | awk '{print $1}' | sort -n > "$temp_sizes"
    
    total_genes=$(wc -l < "$temp_sizes")
    min_sites=$(head -1 "$temp_sizes")
    max_sites=$(tail -1 "$temp_sizes")
    median_sites=$(sort -n "$temp_sizes" | awk '{a[NR]=$1} END {print (NR%2==1) ? a[int(NR/2)+1] : (a[NR/2] + a[NR/2+1])/2}')
    mean_sites=$(awk '{sum+=$1} END {print sum/NR}' "$temp_sizes")
    
    echo "Gene Size Statistics:"
    echo "  Minimum splice sites per gene: $min_sites"
    echo "  Maximum splice sites per gene: $max_sites"
    echo "  Median splice sites per gene:  $median_sites"
    echo "  Mean splice sites per gene:    $(printf "%.1f" $mean_sites)"
    echo "  Total genes analyzed:          $total_genes"
    
    rm "$temp_sizes"
    
    echo ""
    echo "CHROMOSOME-SPECIFIC GENE ANALYSIS"
    echo "--------------------------------"
    echo "Gene density and characteristics by chromosome:"
    echo ""
    
    # Chromosome-specific analysis
    echo "Chromosome | Genes | Avg Sites/Gene | Max Sites | Top Gene"
    echo "-----------+-------+----------------+-----------+-------------------"
    
    for chrom in $(tail -n +2 "$SPLICE_FILE" | cut -f1 | sort -u | sort -V); do
        chrom_data=$(tail -n +2 "$SPLICE_FILE" | awk -v chr="$chrom" '$1==chr')
        
        if [ -n "$chrom_data" ]; then
            gene_count=$(echo "$chrom_data" | cut -f7 | sort -u | wc -l)
            total_sites=$(echo "$chrom_data" | wc -l)
            avg_sites=$(echo "scale=1; $total_sites / $gene_count" | bc -l)
            
            # Find gene with most sites on this chromosome
            top_gene_info=$(echo "$chrom_data" | cut -f7 | sort | uniq -c | sort -nr | head -1)
            max_sites=$(echo $top_gene_info | awk '{print $1}')
            top_gene=$(echo $top_gene_info | awk '{print $2}')
            
            printf "Chr %-6s | %5d | %14s | %9s | %-17s\n" "$chrom" "$gene_count" "$avg_sites" "$max_sites" "$top_gene"
        fi
    done
    
} > "$gene_report"

echo "✅ Gene pattern analysis complete!"
echo "📄 Detailed report saved to: $gene_report"

# Show preview of key findings
echo ""
echo "🔍 KEY FINDINGS PREVIEW:"
echo "========================"

echo ""
echo "🏆 TOP 10 GENES WITH MOST ISOFORMS:"
temp_isoforms=$(mktemp)
tail -n +2 "$SPLICE_FILE" | awk '{print $7 "\t" $8}' | sort -u | \
awk '{count[$1]++} END {for (gene in count) print count[gene], gene}' | \
sort -nr | head -10 > "$temp_isoforms"

while read isoform_count gene_id; do
    splice_count=$(tail -n +2 "$SPLICE_FILE" | awk -v gene="$gene_id" '$7==gene' | wc -l)
    printf "%-17s: %3d isoforms, %4d splice sites\n" "$gene_id" "$isoform_count" "$splice_count"
done < "$temp_isoforms"
rm "$temp_isoforms"

echo ""
echo "📊 SPLICE SITE COMPLEXITY DISTRIBUTION:"
temp_complexity=$(mktemp)
tail -n +2 "$SPLICE_FILE" | cut -f7 | sort | uniq -c | awk '{print $1}' > "$temp_complexity"

total_genes=$(wc -l < "$temp_complexity")
echo "  Simple genes (≤10 sites):    $(awk '$1<=10' "$temp_complexity" | wc -l) genes ($(echo "scale=1; $(awk '$1<=10' "$temp_complexity" | wc -l) * 100 / $total_genes" | bc -l)%)"
echo "  Moderate genes (11-100 sites): $(awk '$1>10 && $1<=100' "$temp_complexity" | wc -l) genes ($(echo "scale=1; $(awk '$1>10 && $1<=100' "$temp_complexity" | wc -l) * 100 / $total_genes" | bc -l)%)"
echo "  Complex genes (>100 sites):  $(awk '$1>100' "$temp_complexity" | wc -l) genes ($(echo "scale=1; $(awk '$1>100' "$temp_complexity" | wc -l) * 100 / $total_genes" | bc -l)%)"

rm "$temp_complexity"

echo ""
echo "📄 Full analysis saved to: $gene_report"