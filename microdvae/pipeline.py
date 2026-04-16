from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .checkpoint import load_microdvae_checkpoint
from .fasta import read_protein_fasta
from .pair_esm import PairESMEmbedder


def _chunk_tensor(x: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
    return [x[start:start + chunk_size] for start in range(0, x.size(0), chunk_size)]


def tokenize_protein_fasta(
    input_fasta: str,
    checkpoint: str,
    output_dir: str,
    pair_esm_model: str = 'h4duan/PAIR-esm2',
    batch_size: int = 32,
    window_size: int = 1024,
    device: str = 'auto',
    esm_dtype: str = 'auto',
    max_length: int = 1024,
) -> Dict[str, object]:
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if window_size <= 0:
        raise ValueError('window_size must be positive')
    if max_length <= 0:
        raise ValueError('max_length must be positive')

    records = read_protein_fasta(input_fasta)
    embedder = PairESMEmbedder.from_pretrained(pair_esm_model, device=device, dtype=esm_dtype)
    protein_embeddings = embedder.embed_records(records, batch_size=batch_size, max_length=max_length)

    model, hyper_parameters = load_microdvae_checkpoint(checkpoint, device=device)
    model_device = next(model.parameters()).device

    token_chunks: List[torch.Tensor] = []
    codebook_chunks: List[torch.Tensor] = []
    for chunk in _chunk_tensor(protein_embeddings, window_size):
        x = chunk.unsqueeze(0).to(model_device)
        mask = torch.ones((1, chunk.size(0)), dtype=torch.long, device=model_device)
        token_ids = model.encode_tokens(x, mask).squeeze(0).cpu()
        codebook_vectors = model.lookup_codebook(token_ids.to(model_device)).cpu()
        token_chunks.append(token_ids)
        codebook_chunks.append(codebook_vectors)

    tokens = torch.cat(token_chunks, dim=0).numpy().astype(np.int64)
    codebook_embeddings = torch.cat(codebook_chunks, dim=0).numpy().astype(np.float32)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / 'tokens.npy', tokens)
    np.save(output_path / 'codebook_embeddings.npy', codebook_embeddings)

    with (output_path / 'tokens.tsv').open('w', encoding='utf-8') as handle:
        handle.write('index\tsequence_id\tdescription\tlength_aa\ttoken_id\n')
        for idx, (record, token_id) in enumerate(zip(records, tokens.tolist())):
            description = record.description.replace('\t', ' ')
            handle.write(f'{idx}\t{record.sequence_id}\t{description}\t{len(record.sequence)}\t{token_id}\n')

    metadata = {
        'input_fasta': str(input_fasta),
        'checkpoint': str(checkpoint),
        'pair_esm_model': pair_esm_model,
        'num_proteins': len(records),
        'embedding_dim': int(protein_embeddings.size(1)),
        'window_size': int(window_size),
        'batch_size': int(batch_size),
        'max_length': int(max_length),
        'device': str(model_device),
        'esm_dtype': esm_dtype,
        'token_shape': list(tokens.shape),
        'codebook_embedding_shape': list(codebook_embeddings.shape),
        'code_dim': int(codebook_embeddings.shape[1]),
        'sequence_ids': [record.sequence_id for record in records],
        'hyper_parameters': hyper_parameters,
    }
    with (output_path / 'metadata.json').open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

    return metadata
