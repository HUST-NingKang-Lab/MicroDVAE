#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from microdvae.pipeline import tokenize_protein_fasta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Tokenize an ordered protein FASTA with MicroDVAE and export token IDs plus codebook embeddings.')
    parser.add_argument('--input', required=True, help='Protein FASTA file in genomic order.')
    parser.add_argument('--checkpoint', required=True, help='Path to the MicroDVAE checkpoint file.')
    parser.add_argument('--output-dir', required=True, help='Directory for tokens and embeddings.')
    parser.add_argument('--pair-esm-model', default='h4duan/PAIR-esm2', help='PAIR-esm2 Hugging Face model ID or local path.')
    parser.add_argument('--batch-size', type=int, default=32, help='PAIR-esm2 inference batch size.')
    parser.add_argument('--window-size', type=int, default=1024, help='Number of proteins per dVAE chunk.')
    parser.add_argument('--device', default='auto', help='Torch device, for example auto, cpu, cuda, or cuda:0.')
    parser.add_argument('--esm-dtype', default='auto', choices=['auto', 'float32', 'float16', 'bfloat16'], help='Data type used for PAIR-esm2 inference.')
    parser.add_argument('--max-length', type=int, default=1024, help='Maximum amino-acid length per protein passed to PAIR-esm2.')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metadata = tokenize_protein_fasta(
        input_fasta=args.input,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        pair_esm_model=args.pair_esm_model,
        batch_size=args.batch_size,
        window_size=args.window_size,
        device=args.device,
        esm_dtype=args.esm_dtype,
        max_length=args.max_length,
    )
    print(f"Processed {metadata['num_proteins']} proteins")
    print(f"Tokens saved to {Path(args.output_dir) / 'tokens.npy'}")
    print(f"Codebook embeddings saved to {Path(args.output_dir) / 'codebook_embeddings.npy'}")


if __name__ == '__main__':
    main()
