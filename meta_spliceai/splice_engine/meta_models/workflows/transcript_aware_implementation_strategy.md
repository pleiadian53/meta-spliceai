# Transcript-Aware Position Identification: Implementation Strategy

## Phase 1: Preserve Transcript Information (Current)
- Keep current genomic-only grouping for ML efficiency
- Add `transcript_ids` column as a list/set for each position
- Add `transcript_count` to track how many isoforms use each site
- This maintains backward compatibility while preserving information

## Phase 2: Hybrid Analysis Mode
- Add a `--transcript-aware` flag to the workflow
- When enabled:
  - Perform initial predictions at genomic level
  - Then expand to transcript-specific predictions
  - Report both aggregated and transcript-specific metrics

## Phase 3: Full Transcript-Specific Mode
- Complete transcript-level predictions
- Useful for:
  - Novel isoform discovery
  - Clinical interpretation
  - Tissue-specific splicing analysis

## Configuration Example

```python
class PositionIdentificationConfig:
    def __init__(
        self,
        mode='genomic',  # 'genomic', 'transcript', 'hybrid'
        preserve_transcript_info=True,
        group_by_columns=None,
        conflict_resolution='priority',  # 'priority', 'vote', 'max_score'
        aggregate_transcript_scores=True
    ):
        self.mode = mode
        self.preserve_transcript_info = preserve_transcript_info
        self.group_by_columns = group_by_columns or self._get_default_columns()
        self.conflict_resolution = conflict_resolution
        self.aggregate_transcript_scores = aggregate_transcript_scores
    
    def _get_default_columns(self):
        if self.mode == 'genomic':
            return ['gene_id', 'position', 'strand']
        elif self.mode == 'transcript':
            return ['gene_id', 'position', 'strand', 'transcript_id']
        elif self.mode == 'hybrid':
            return ['gene_id', 'position', 'strand']  # But preserve transcript info
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
```

## Backward Compatibility

To maintain compatibility with existing workflows:

1. Default to current behavior (`mode='genomic'`)
2. Add optional transcript preservation without changing output format
3. Provide migration utilities for existing models
4. Document the biological implications clearly

## Benefits of This Approach

1. **Immediate**: Better biological understanding without breaking changes
2. **Short-term**: Improved debugging and analysis capabilities  
3. **Long-term**: Foundation for isoform-specific predictions
4. **Clinical**: Better support for precision medicine applications
