import meta_spliceai.sphere_pipeline.data_model as dm
# NOTE: For now, just borrowing the data model from the sphere pipeline


class TranscriptomicSequence(dm.TranscriptSequenceIO):  
    artifact_dir = dataset_name = "sequence"
    # data_prefix = DataSource.data_prefix # os.path.join(os.getcwd(), dataset_name)

    # Training data root directories
    sequence_dir = 'sequence'
    marker_dir = 'marker'

    source_columns = ['sequence', 'marker', ]
    col_seq = 'sequence'
    col_marker = 'marker'

# Alias for the TranscriptomicSequence class
Sequence = TranscriptomicSequence


class SequenceCodeBook: 
    char_oov = unk = 'N'
    char_padding = 'Z'
    voc2int = {voc:ind for ind,voc in enumerate([char_padding, unk, 'A', 'T', 'C', 'G'])}

    @staticmethod
    def get_codebook(): 
        return SequenceCodeBook.voc2int


class SequenceMarkers(dm.SequenceMarkers): 
    markers = {'intron': '-',
               'exon': 'x',
               'coding_exon': 'e',  # CDS marker

               # Reserved for future use
               # 'start_codon': 'p', 'candidate_start_codon': 'u',
               # 'stop_codon': 'q', 'candidate_stop_codon': 'v', 
               # 'five_prime_utr': '5', 'three_prime_utr': '3', 
               # 'selenocysteine': 'y',
            }
    # NOTE: Noncoding exon and coding exon was orginally distingusihed by case (e.g. 'e' vs 'E')
    #       but this may make it harder to work with Spark due to its default behavior being case-insensitive

    char_oov = unk = 'N'
    char_padding = 'Z'
    voc2int = {}

    @staticmethod
    def get_codebook(): 
        padding = SequenceMarkers.char_padding
        unk = SequenceMarkers.unk
        ftypes = ['intron', 'exon', 'coding_exon', # 2, 3, 4
                    'five_prime_utr', 'three_prime_utr', # 5, 6
                    'start_codon', 'stop_codon', # 7, 8
                        'candidate_start_codon', 'candidate_stop_codon', 
                        'selenocysteine', 
                ]
        other_markers = [padding, unk]
        markers = other_markers + [SequenceMarkers.markers[ftype] for ftype in ftypes]
        voc2int = {voc:ind for ind,voc in enumerate(markers)}
        return voc2int

