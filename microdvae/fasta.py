from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from Bio import SeqIO


@dataclass(frozen=True)
class ProteinRecord:
    index: int
    sequence_id: str
    description: str
    sequence: str


def read_protein_fasta(path: str | Path) -> List[ProteinRecord]:
    fasta_path = Path(path)
    if not fasta_path.exists():
        raise FileNotFoundError(f'Input FASTA not found: {fasta_path}')

    records: List[ProteinRecord] = []
    for idx, record in enumerate(SeqIO.parse(str(fasta_path), 'fasta')):
        sequence = str(record.seq).strip()
        if not sequence:
            continue
        records.append(
            ProteinRecord(
                index=idx,
                sequence_id=record.id,
                description=record.description,
                sequence=sequence,
            )
        )

    if not records:
        raise ValueError(f'No non-empty protein sequences were found in {fasta_path}')
    return records
