"""
Data source ingesters for splice mutation databases.

This module provides classes to ingest and standardize data from various
splice mutation databases including SpliceVarDB, MutSpliceDB, DBASS, and ClinVar.
"""

from .splicevardb import SpliceVarDBIngester
from .mutsplicedb import MutSpliceDBIngester  
from .dbass import DBASSIngester
from .clinvar import ClinVarIngester
from .base import BaseIngester, IngestionResult

__all__ = [
    "SpliceVarDBIngester",
    "MutSpliceDBIngester",
    "DBASSIngester", 
    "ClinVarIngester",
    "BaseIngester",
    "IngestionResult"
] 