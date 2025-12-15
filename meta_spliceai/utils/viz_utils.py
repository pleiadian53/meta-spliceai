import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collections import defaultdict
from .bio_utils import parse_gtf

##### Visualization with Matplotlib

# Assume `features` is the list of dictionaries returned by `parse_gtf(gtf_file_path)`.
def extract_transcript_features(features):
    """
    Organizes the parsed GTF data into a structure that groups exons, CDS regions, 
    and splice sites by transcript ID, assuming that 
    `features` is the list of dictionaries returned by `parse_gtf(gtf_file_path)`.

    Parameters: 
    features (list): List of dictionaries containing parsed GTF data.

    Example usage:
    transcript_data = extract_transcript_features(features)
    """
    transcript_data = {}
    splice_site_summary = {'with_splice_sites': [], 'without_splice_sites': []}
    
    for feature in features:
        transcript_id = feature['attributes'].get('transcript_id')
        if not transcript_id:
            continue
        
        if transcript_id not in transcript_data:
            transcript_data[transcript_id] = {'exons': [], 'cds': [], 'splice_sites': []}
        
        start = feature['start']
        end = feature['end']
        feature_type = feature['feature']
        
        # Extract exons
        if feature_type == 'exon':
            transcript_data[transcript_id]['exons'].append((start, end))
        
        # Extract CDS
        elif feature_type == 'CDS':
            transcript_data[transcript_id]['cds'].append((start, end))
        
        # Extract splice sites if present
        if 'splice sites' in feature['attributes']:
            splice_sites = feature['attributes']['splice sites']
            # print(f"[test] Extracted Splice Sites: {splice_sites}")  
            transcript_data[transcript_id]['splice_sites'].append(splice_sites)
        # else: 
        #     print(f"[test] No Splice Sites Found for Transcript: {transcript_id}")
        #     print(feature['attributes'])

    # Check for splice site annotations
    for transcript_id, data in transcript_data.items():
        if data['splice_sites']:
            splice_site_summary['with_splice_sites'].append(transcript_id)
        else:
            splice_site_summary['without_splice_sites'].append(transcript_id)
    
    return transcript_data, splice_site_summary


def save_plot(fig, save_path, file_format='pdf', verbose=0):
    """
    Saves the plot to the specified path and format.

    Parameters:
    - fig: matplotlib.figure.Figure, the figure object to save
    - save_path: str, the path to save the plot
    - file_format: str, the format to save the plot (default is 'pdf')
    """
    # Check if save_path already includes a file extension
    base, ext = os.path.splitext(save_path)
    if ext:
        ext = ext.lstrip('.')
        if ext != file_format:
            print(f"Warning: The file extension in save_path ({ext}) does not match the specified file_format ({file_format}).")
        # Save the plot using the extension in save_path
        fig.savefig(save_path)
    else:
        # Save the plot using the specified file_format
        fig.savefig(f'{save_path}.{file_format}', format=file_format)
    
    if verbose:
        print(f"[i/o] Plot saved to: {save_path}.{file_format}")


def plot_transcript_v0(transcript_data, transcript_id, save_path, file_format='pdf', verbose=0):
    """
    Plots the transcript data and saves it to the specified path and format.
    
    Exons are drawn as greenish rectangles.
    CDS regions are highlighted within exons in pinkish colors.
    Splice sites are marked with dark purple lines and circles.

    Parameters:
    - transcript_data: dict, the transcript data
    - transcript_id: str, the ID of the transcript to plot
    - save_path: str, the path to save the plot
    - file_format: str, the format to save the plot (default is 'pdf')

    Example usage:
    plot_transcript(transcript_data, 'uorft_denovo_2', 'output/transcript_plot', 'pdf')

    Memo:

    * Matplotlib supports the following file formats:
        - 'pdf' (default)
        - 'png' (high-quality raster image)
        - 'svg' (scalable vector graphics)
        - 'eps' (encapsulated postscript)
        - 'jpg' or 'jpeg' (lossy compressed image)
    """

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))

    # Get the exons, CDS, and splice sites for the transcript
    exons = transcript_data[transcript_id]['exons']
    cds = transcript_data[transcript_id]['cds']
    splice_sites = transcript_data[transcript_id]['splice_sites']

    # Plot introns as lines connecting exons
    for i in range(1, len(exons)):
        intron_start = exons[i-1][1]
        intron_end = exons[i][0]
        ax.plot([intron_start, intron_end], [0.5, 0.5], color='black', linestyle='-', linewidth=1)

    # Plot each exon as a greenish block
    for exon_start, exon_end in exons:
        ax.add_patch(patches.Rectangle((exon_start, 0.45), exon_end - exon_start, 0.1, 
                                       edgecolor='black', facecolor='#66C2A5'))

    # Highlight CDS regions within exons in a pinkish color
    for cds_start, cds_end in cds:
        ax.add_patch(patches.Rectangle((cds_start, 0.47), cds_end - cds_start, 0.06, 
                                       edgecolor='black', facecolor='#FC8D62'))

    # Plot splice sites as arrows indicating splice junctions
    for ss_start, ss_end in splice_sites:
        ax.annotate('', xy=(ss_end, 0.5), xytext=(ss_start, 0.5),
                    arrowprops=dict(arrowstyle="->", color='#6A3D9A', lw=2))

    # Set axis limits and labels
    ax.set_xlim(min([exon[0] for exon in exons]) - 100, max([exon[1] for exon in exons]) + 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Genomic Position')
    ax.set_ylabel('Transcript Features')
    
    # Title with transcript ID
    ax.set_title(f'Transcript: {transcript_id}')
    
    # Save the plot to the specified path and format
    # plt.savefig(f'{save_path}.{file_format}', format=file_format)
    save_plot(fig, save_path, file_format, verbose=verbose)
    plt.close()


def plot_transcript(transcript_data, transcript_id, save_path, file_format='pdf', verbose=0):
    """
    Plots the transcript data and saves it to the specified path and format.
    
    Exons are drawn as greenish blocks.
    Splice sites are marked with dark purple lines.

    Parameters:
    - transcript_data: dict, the transcript data
    - transcript_id: str, the ID of the transcript to plot
    - save_path: str, the path to save the plot
    - file_format: str, the format to save the plot (default is 'pdf')
    """

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))

    # Get the exons, CDS, splice sites, and strand for the transcript
    exons = transcript_data[transcript_id]['exons']
    cds = transcript_data[transcript_id]['cds']
    splice_sites = transcript_data[transcript_id]['splice_sites']
    strand = transcript_data[transcript_id].get('strand', '+')  # Default to '+' if strand info is missing

    # Plot introns as lines connecting exons
    for i in range(1, len(exons)):
        intron_start = exons[i-1][1]
        intron_end = exons[i][0]
        ax.plot([intron_start, intron_end], [0.5, 0.5], color='black', linestyle='-', linewidth=1)

    # Plot each exon as a greenish block
    for exon_start, exon_end in exons:
        ax.add_patch(patches.Rectangle((exon_start, 0.45), exon_end - exon_start, 0.1, 
                                       edgecolor='black', facecolor='#66C2A5'))

    # Plot splice sites with directional arrows based on the strand
    for ss_start, ss_end in splice_sites:
        if strand == '+':
            ax.annotate('', xy=(ss_end, 0.5), xytext=(ss_start, 0.5),
                        arrowprops=dict(arrowstyle="->", color='#FC8D62', lw=2)) # pinkish: '#FC8D62', dark purple: '#6A3D9A'
        else:  # strand == '-'
            ax.annotate('', xy=(ss_start, 0.5), xytext=(ss_end, 0.5),
                        arrowprops=dict(arrowstyle="->", color='#FC8D62', lw=2))
    
    # Set axis limits and labels
    ax.set_xlim(min([exon[0] for exon in exons]) - 100, max([exon[1] for exon in exons]) + 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Genomic Position')
    ax.set_ylabel('Transcript Features')
    
    # Title with transcript ID
    ax.set_title(f'Transcript: {transcript_id} (strand: {strand})')
    
    # Save the plot to the specified path and format
    plt.savefig(f'{save_path}.{file_format}', format=file_format)
    if verbose: print(f"[i/o] Plot saved to: {save_path}.{file_format}")

    plt.close()


def plot_multiple_transcripts(transcript_data, output_dir, num_transcripts=10, file_format='pdf', verbose=1):
    """
    Plots multiple transcripts and saves them to the specified output directory.
    
    Parameters:
    - transcript_data: dict, the transcript data
    - output_dir: str, the directory to save the plots
    - num_transcripts: int, the number of transcripts to plot (default is 10)
    - file_format: str, the format to save the plots (default is 'pdf')

    Example usage:
    
    output_directory = 'output/transcript_plots'
    plot_multiple_transcripts(transcript_data, output_directory, num_transcripts=10, file_format='pdf')

    """

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Select the transcript IDs (randomly or by some other criteria)
    selected_transcripts = random.sample(list(transcript_data.keys()), num_transcripts)

    # Plot each selected transcript
    for transcript_id in selected_transcripts:
        # Construct the file path for saving
        save_path = os.path.join(output_dir, f'{transcript_id}_plot')
        
        # Plot and save the transcript
        plot_transcript(transcript_data, transcript_id, save_path, file_format, verbose=verbose)


def gtf_to_bed_v0(gtf_file_path, bed_file_path):
    """
    Convert a GTF file to BED format using the parsed features from the GTF file.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.
    - bed_file_path (str): Path to the output BED file.

    Example usage: 

    gtf_to_bed('path/to/input.gtf', 'path/to/output.bed')
    """
    # Processes the GTF file and extracts all relevant features into a list of dictionaries 
    features = parse_gtf(gtf_file_path) 
    
    bed_lines = []

    for feature in features:
        chrom = feature['seqname']
        start = feature['start'] - 1  # BED format uses 0-based start
        end = feature['end']  # BED uses 1-based end
        name = feature['attributes'].get('transcript_id', 'NA')  # Default to 'NA' if no transcript_id
        score = feature['score'] if feature['score'] != '.' else '0'
        strand = feature['strand']

        # Use color coding based on feature type
        if feature['feature'] == 'exon':
            item_rgb = '0,255,0'  # Greenish for exons
        elif feature['feature'] == 'CDS':
            item_rgb = '255,192,203'  # Pinkish for CDS regions
        else:
            item_rgb = '0,0,0'  # Default color for other features

        # Handle splice sites separately, if they exist
        if 'splice sites' in feature['attributes']:
            splice_sites = feature['attributes']['splice sites']
            # print(f"> splice sites (type: {type(splice_sites)}): {splice_sites}")
            # for (ss_start, ss_end) in splice_sites:
            assert isinstance(splice_sites, tuple) and isinstance(splice_sites[0], int)

            ss_start, ss_end = splice_sites
            bed_line = f"{chrom}\t{ss_start - 1}\t{ss_end}\t{name}_splice\t{score}\t{strand}\t{ss_start - 1}\t{ss_end}\t{item_rgb}"
            bed_lines.append(bed_line)

            # NOTE: The splice site coordinates are used as the start and end positions in the BED file
            #  - name_splice: 
            #    This is the name of the transcript (as extracted from the GTF attribute field) 
            #    with the suffix "_splice" to indicate that this line represents a splice junction.
            #  - `ss_start - 1` and `ss_end (2nd occurrence)`: 
            #     These are the positions for thickStart and thickEnd in BED format, 
            #     which can be used to highlight a specific region of the feature. 
            #     In this case, it's set to highlight the entire splice site.
            #  - item_rgb: 
            #    This defines the color of the feature in the genome browser. 
            #    Here, it indicates the color assigned to splice sites.
        else:
            bed_line = f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\t{start}\t{end}\t{item_rgb}"
            bed_lines.append(bed_line)

    # Write to BED file
    print("[i/o] Writing BED file to:", bed_file_path)
    with open(bed_file_path, 'w') as bed_file:
        for line in bed_lines:
            bed_file.write(line + '\n')

    return


def gtf_to_bed(gtf_file_path, bed_file_path, exon_bed_file_path=None, cds_bed_file_path=None, 
               exon_color="144,238,144", cds_color="255,182,193", junction_color="173,216,230"):
    """
    Convert a GTF file to BED format using the parsed features from the GTF file.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.
    - bed_file_path (str): Path to the output BED file.
    - exon_bed_file_path (str): Path to the output BED file for exons (optional).
    - cds_bed_file_path (str): Path to the output BED file for CDS (optional).
    - exon_color (str): RGB color string for exons (default: "0,255,0" - green).
    - cds_color (str): RGB color string for CDS (default: "255,192,203" - pink).
    - junction_color (str): RGB color string for junctions (default: "0,0,255" - blue).
    
    Example usage:
    gtf_to_bed('path/to/input.gtf', 'path/to/output.bed', exon_bed_file_path='path/to/exons.bed', 
               cds_bed_file_path='path/to/cds.bed', exon_color="0,128,0", cds_color="255,105,180")

    Memo: 
        Pastel color scheme
        pastel_blue = "173,216,230"  # Pastel blue for junctions
        pastel_green = "144,238,144"  # Pastel green for exons
        pastel_pink = "255,182,193"  # Pastel pink for CDS (other hues: "255,192,203", "255,182,193")
    """
    # Processes the GTF file and extracts all relevant features into a list of dictionaries 
    gtf_features = parse_gtf(gtf_file_path) 
    
    bed_lines = []
    bed12_lines = []
    exon_bed_lines = []
    cds_bed_lines = []

    processed_junctions = set()  # Track processed junctions to avoid duplication
    
    for feature in gtf_features:
        chrom = feature['seqname']
        start = feature['start'] - 1  # BED is 0-based
        end = feature['end']
        strand = feature['strand']
        attributes = feature['attributes']
        name = attributes.get('transcript_id', 'unknown')

        # Handle splice junctions
        if 'splice sites' in attributes:
            ss_start, ss_end = attributes['splice sites']
            junction_name = f"{name}_jnct"  # Preserve original ID, shorten suffix
            
            # Check if the junction has already been processed
            if (chrom, ss_start, ss_end, strand) not in processed_junctions:
                processed_junctions.add((chrom, ss_start, ss_end, strand))
                # Optionally, add thin line width specification (if supported by IGV in custom track)
                # bed_line = f"{chrom}\t{ss_start - 1}\t{ss_end}\t{junction_name}\t0\t{strand}\t{ss_start - 1}\t{ss_end}\t{junction_color}\tthin"
                bed_line = f"{chrom}\t{ss_start - 1}\t{ss_end}\t{junction_name}\t0\t{strand}\t{ss_start - 1}\t{ss_end}\t{junction_color}"
                bed_lines.append(bed_line)
                
        # Handle exons
        if feature['feature'] == 'exon':
            exon_name = f"{name}_ex"  # Preserve original ID, shorten suffix
            bed_line = f"{chrom}\t{start}\t{end}\t{exon_name}\t0\t{strand}\t{start}\t{end}\t{exon_color}"
            bed_lines.append(bed_line)
            
            # Additionally, add to exon BED file
            exon_bed_lines.append(bed_line)
        
        # Handle CDS
        elif feature['feature'] == 'CDS':
            cds_name = f"{name}_cds"  # Preserve original ID, shorten suffix
            bed_line = f"{chrom}\t{start}\t{end}\t{cds_name}\t0\t{strand}\t{start}\t{end}\t{cds_color}"
            bed_lines.append(bed_line)

            # Additionally, add to CDS BED file
            cds_bed_lines.append(bed_line)
    
    # Write to BED file
    print("[i/o] Writing combined BED file to:", bed_file_path)
    with open(bed_file_path, 'w') as bed_file:
        for line in bed_lines:
            bed_file.write(line + '\n')

    if exon_bed_file_path: 
        print("[i/o] Writing exon BED file to:", exon_bed_file_path)
        with open(exon_bed_file_path, 'w') as exon_bed_file:
            for line in exon_bed_lines:
                exon_bed_file.write(line + '\n') 
    
    if cds_bed_file_path:
        print("[i/o] Writing CDS BED file to:", cds_bed_file_path)
        with open(cds_bed_file_path, 'w') as cds_bed_file:
            for line in cds_bed_lines:
                cds_bed_file.write(line + '\n')
    
    return


def gtf_to_bed12(gtf_file_path, bed_file_path):
    """
    Convert a GTF file to a BED12 file for better visualization of exons and splice junctions.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.
    - bed_file_path (str): Path to the output BED file.
    """
    gtf_features = parse_gtf(gtf_file_path)
    
    bed12_lines = []
    
    for feature in gtf_features:
        chrom = feature['seqname']
        start = feature['start'] - 1  # BED is 0-based
        end = feature['end']
        strand = feature['strand']
        attributes = feature['attributes']
        transcript_id = attributes.get('transcript_id', 'unknown')
        gene_id = attributes.get('gene_id', 'unknown')

        if feature['feature'] == 'exon':
            exon_name = f"{transcript_id}_exon_{gene_id}"
            bed12_lines.append(
                f"{chrom}\t{start}\t{end}\t{exon_name}\t0\t{strand}\t{start}\t{end}\t0,255,0\t1\t{end-start}\t0"
            )
        
        elif feature['feature'] == 'splice_site':
            splice_name = f"{transcript_id}_splice_{gene_id}"
            block_sizes = f"{end-start},100"
            block_starts = f"0,{end-start+100}"
            bed12_lines.append(
                f"{chrom}\t{start}\t{end+100}\t{splice_name}\t0\t{strand}\t{start}\t{end}\t255,192,203\t2\t{block_sizes}\t{block_starts}"
            )

    # Write BED12 file
    print(f"[i/o] Writing BED12 file to: {bed_file_path}")
    with open(bed_file_path, 'w') as bed_file:
        for line in bed12_lines:
            bed_file.write(line + '\n')

    return


def gtf_to_exon_and_cds_bed(gtf_file_path, exon_bed_file_path, cds_bed_file_path):
    """
    Convert a GTF file to two separate BED files for exons and CDS regions.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.
    - exon_bed_file_path (str): Path to the output BED file for exons.
    - cds_bed_file_path (str): Path to the output BED file for CDS regions.
    """
    gtf_features = parse_gtf(gtf_file_path)
    
    exon_bed_lines = []
    cds_bed_lines = []
    
    for feature in gtf_features:
        chrom = feature['seqname']
        start = feature['start'] - 1  # BED is 0-based
        end = feature['end']
        strand = feature['strand']
        attributes = feature['attributes']
        name = attributes.get('transcript_id', 'unknown')
        short_name = name.split('_')[-1]  # Extract a shorter name, e.g., 'u5'

        if feature['feature'] == 'exon':
            exon_name = f"{name}_ex"
            exon_bed_lines.append(
                f"{chrom}\t{start}\t{end}\t{exon_name}\t0\t{strand}\t{start}\t{end}\t0,255,0"
            )
        
        elif feature['feature'] == 'CDS':
            cds_name = f"{name}_cds"
            cds_bed_lines.append(
                f"{chrom}\t{start}\t{end}\t{cds_name}\t0\t{strand}\t{start}\t{end}\t255,192,203"
            )

    # Write exon BED file
    print(f"[i/o] Writing exon BED file to: {exon_bed_file_path}")
    with open(exon_bed_file_path, 'w') as exon_bed_file:
        for line in exon_bed_lines:
            exon_bed_file.write(line + '\n')

    # Write CDS BED file
    print(f"[i/o] Writing CDS BED file to: {cds_bed_file_path}")
    with open(cds_bed_file_path, 'w') as cds_bed_file:
        for line in cds_bed_lines:
            cds_bed_file.write(line + '\n')

    return



def plot_transcript_structure(transcript_structure, transcript_id):
    """

    Usage example:

    # Plot the structure of the given transcript
    plot_transcript_structure(transcript_structure, "uorft_denovo_2")
    """

    exons = transcript_structure[transcript_id]['exons']
    cds = transcript_structure[transcript_id]['cds']

    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Plot exons as boxes
    for start, end in exons:
        ax.add_patch(plt.Rectangle((start, 0.4), end-start, 0.2, color='blue', label='Exon'))
    
    # Plot CDS as thicker boxes
    for start, end in cds:
        ax.add_patch(plt.Rectangle((start, 0.6), end-start, 0.4, color='green', label='CDS'))
    
    # Annotate start and end of the transcript
    ax.annotate("5'", (exons[0][0], 0.75), verticalalignment='center')
    ax.annotate("3'", (exons[-1][1], 0.75), verticalalignment='center')
    
    ax.set_xlim(min(exons[0][0], cds[0][0]), max(exons[-1][1], cds[-1][1]))
    ax.set_ylim(0, 1)
    
    ax.set_yticks([])
    ax.set_xlabel('Genomic Position')
    ax.set_title(f'Transcript Structure: {transcript_id}')
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.show()


def display_splice_site_summary(splice_site_summary, verbose=0):
    """
    Displays the splice site summary in an easy-to-read manner.

    Parameters:
    - splice_site_summary (dict): A dictionary with keys 'with_splice_sites' and 'without_splice_sites'
      containing lists of transcript IDs.

    Example usage:
    display_splice_site_summary(splice_site_summary)
    """
    with_splice_sites = splice_site_summary.get('with_splice_sites', [])
    without_splice_sites = splice_site_summary.get('without_splice_sites', [])

    print("Splice Site Summary:")
    print("====================")

    print(f"Transcripts with splice sites: {len(with_splice_sites)}")
    if verbose: 
        for transcript_id in with_splice_sites:
            print(f"  - {transcript_id}")

    print(f"\nTranscripts without splice sites: {len(without_splice_sites)}")
    if verbose: 
        for transcript_id in without_splice_sites:
            print(f"  - {transcript_id}")


def demo_plot_transcript_features_matplotlib():  
    
    data_path = "/path/to/meta-spliceai/data/ensembl/"
    plot_path = os.path.join(data_path, "plot")
    uorf_tx_gtf = os.path.join(data_path, "h38.final.gtf")  # "/path/to/meta-spliceai/data/ensembl/h38.final.gt"

    # Create the directory if it doesn't exist
    os.makedirs(plot_path, exist_ok=True)

    features = parse_gtf(uorf_tx_gtf)  # Parse GTF file into a list of dictionaries
    transcript_data, splice_site_summary = \
        extract_transcript_features(features)  # Extract transcript features

    num_transcripts = len(transcript_data)
    print(f"[info] Total number of distinct uORF-connected transcripts: {num_transcripts}")

    # Display splice site summary
    display_splice_site_summary(splice_site_summary)

    # plot_transcript(transcript_data, transcript_id, save_path, file_format='pdf')
    plot_multiple_transcripts(transcript_data, plot_path, num_transcripts=10, file_format='pdf', verbose=1)

    return

def demo_plot_transcript_features_igv(): 
    """

    Memo: 
    - /path/to/meta-spliceai/data/ensembl/h38.final.gtf
    - /path/to/meta-spliceai/data/ensembl/h38_dup_removed.final.gtf
    """

    data_path = "/path/to/meta-spliceai/data/ensembl/"
    plot_path = os.path.join(data_path, "plot")
    uorf_tx_gtf = os.path.join(data_path, "h38_dup_removed.final.gtf")  # "/path/to/meta-spliceai/data/ensembl/h38.final.gt"
    uorf_tx_bed = os.path.join(data_path, "h38_dup_removed.final.bed")
    uorf_exon_bed = os.path.join(data_path, "h38_dup_removed.final.exon.bed")
    uorf_cds_bed = os.path.join(data_path, "h38_dup_removed.final.cds.bed")

    # Create the directory if it doesn't exist
    os.makedirs(plot_path, exist_ok=True)

    print("[info] Converting GTF to BED format...")
    gtf_to_bed(uorf_tx_gtf, uorf_tx_bed, 
               exon_bed_file_path=uorf_exon_bed, 
               cds_bed_file_path=uorf_cds_bed)

    return

def demo_parse_gtf(): 

    # Sample GTF data as a list of strings
    gtf_data = """
    1       uORFexplorer    transcript      100352597       100518280       .       +       .       gene_id "ENSG00000079335"; orf_id "c1riboseqorf129"; transcript_id "uorft_denovo_2"; reference_id "ENST00000336454"; splice sites (100352645, 100353761);
    1       uORFexplorer    exon    100352597       100352645       .       +       .       gene_id "ENSG00000079335"; orf_id "c1riboseqorf129"; transcript_id "uorft_denovo_2"; reference_id "ENST00000336454";
    1       uORFexplorer    CDS     100352597       100352645       .       +       .       gene_id "ENSG00000079335"; orf_id "c1riboseqorf129"; transcript_id "uorft_denovo_2"; reference_id "ENST00000336454";
    1       uORFexplorer    exon    100353762       100353852       .       +       .       gene_id "ENSG00000079335"; orf_id "c1riboseqorf129"; transcript_id "uorft_denovo_2"; reference_id "ENST00000336454";
    1       uORFexplorer    CDS     100353762       100353852       .       +       .       gene_id "ENSG00000079335"; orf_id "c1riboseqorf129"; transcript_id "uorft_denovo_2"; reference_id "ENST00000336454";
    1       uORFexplorer    exon    100377546       100377621       .       +       .       gene_id "ENSG00000079335"; orf_id "c1riboseqorf129"; transcript_id "uorft_denovo_2"; reference_id "ENST00000336454";
    1       uORFexplorer    CDS     100377546       100377621       .       +       .       gene_id "ENSG00000079335"; orf_id "c1riboseqorf129"; transcript_id "uorft_denovo_2"; reference_id "ENST00000336454";
    1       uORFexplorer    exon    100390732       100390824       .       +       .       gene_id "ENSG00000079335"; orf_id "c1riboseqorf129"; transcript_id "uorft_denovo_2"; reference_id "ENST00000336454";
    1       uORFexplorer    CDS     100390732       100390824       .       +       .       gene_id "ENSG00000079335"; orf_id "c1riboseqorf129"; transcript_id "uorft_denovo_2"; reference_id "ENST00000336454";
    """

    # Parse GTF data
    transcript_structure = defaultdict(lambda: {'exons': [], 'cds': []})

    for line in gtf_data.strip().split("\n"):
        columns = line.split("\t")
        chrom, source, feature_type, start, end, score, strand, frame, attributes = columns
        start, end = int(start), int(end)
        
        # Extract the transcript ID from attributes
        attrs = {attr.split(" ")[0]: attr.split(" ")[1].replace('"', '') for attr in attributes.split("; ") if attr}
        transcript_id = attrs.get("transcript_id")
        
        if feature_type == "exon":
            transcript_structure[transcript_id]['exons'].append((start, end))
        elif feature_type == "CDS":
            transcript_structure[transcript_id]['cds'].append((start, end))

    # Print parsed structure for debugging
    print(transcript_structure)


def test(): 

    # demo_plot_transcript_features_matplotlib()

    demo_plot_transcript_features_igv()


if __name__ == "__main__":
    test()