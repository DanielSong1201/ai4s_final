"""Cache Hugging Face ESM residue embeddings for the trainable affinity manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STR_ROOT = PROJECT_ROOT / "str"
for import_root in (STR_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.data.create_esm_manifest import display_path, project_path  # noqa: E402


DEFAULT_MANIFEST = Path("str/manifest/esm_affinity_trainable_manifest.csv")
DEFAULT_CACHE_DIR = Path("str/manifest/cache/esm_embeddings")
DEFAULT_REPORT = Path("str/manifest/cache/esm_embeddings_report.json")
DEFAULT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DEFAULT_MAX_RESIDUES_PER_CHUNK = 1022


def progress_bar(iterable, desc: str, unit: str):
    return tqdm(
        iterable,
        desc=desc,
        unit=unit,
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
        disable=False,
        miniters=1,
        mininterval=0.0,
    )


def load_transformers():
    try:
        from transformers import AutoTokenizer, EsmModel
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for Hugging Face ESM caching. "
            "Install it with: pip install transformers accelerate safetensors"
        ) from exc
    return AutoTokenizer, EsmModel


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_chains(sequence: str) -> list[str]:
    chains = [chain.strip().replace(" ", "") for chain in str(sequence).split(":")]
    return [chain for chain in chains if chain]


def chunk_chain(chain: str, max_residues: int) -> list[tuple[int, int, str]]:
    if max_residues <= 0:
        raise ValueError("--max-residues-per-chunk must be positive")
    chunks = []
    for start in range(0, len(chain), max_residues):
        end = min(start + max_residues, len(chain))
        chunks.append((start, end, chain[start:end]))
    return chunks


def batched(values: list[dict[str, object]], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def extract_residue_embeddings(hidden: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
    embeddings = []
    for index in range(hidden.shape[0]):
        valid_len = int(attention_mask[index].sum().item())
        residue_embedding = hidden[index, 1 : valid_len - 1].detach().cpu()
        embeddings.append(residue_embedding)
    return embeddings


def encode_chunks(
    chunks: list[dict[str, object]],
    tokenizer,
    model,
    device: torch.device,
    chunk_batch_size: int,
) -> list[dict[str, object]]:
    encoded_chunks = []
    for chunk_batch in batched(chunks, chunk_batch_size):
        sequences = [str(item["sequence"]) for item in chunk_batch]
        inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        residue_embeddings = extract_residue_embeddings(outputs.last_hidden_state, inputs["attention_mask"])
        for item, embedding in zip(chunk_batch, residue_embeddings):
            expected_len = int(item["end"]) - int(item["start"])
            if embedding.shape[0] != expected_len:
                raise ValueError(
                    f"Chunk length mismatch for {item['pdb_id']} chain {item['chain_index']}: "
                    f"expected {expected_len}, got {embedding.shape[0]}"
                )
            encoded_chunks.append({**item, "embedding": embedding})
    return encoded_chunks


def cache_one_row(row, tokenizer, model, device: torch.device, args: argparse.Namespace) -> dict[str, object]:
    chains = split_chains(row.protein_sequence)
    if not chains:
        raise ValueError(f"No chains found for {row.pdb_id}")

    chunks = []
    chain_lengths = []
    for chain_index, chain in enumerate(chains):
        chain_lengths.append(len(chain))
        for chunk_index, (start, end, chunk_sequence) in enumerate(chunk_chain(chain, args.max_residues_per_chunk)):
            chunks.append(
                {
                    "pdb_id": row.pdb_id,
                    "chain_index": chain_index,
                    "chunk_index": chunk_index,
                    "start": start,
                    "end": end,
                    "sequence": chunk_sequence,
                }
            )

    encoded_chunks = encode_chunks(chunks, tokenizer, model, device, args.chunk_batch_size)
    chain_embeddings = []
    chunk_metadata = []
    for chain_index in range(len(chains)):
        chain_chunks = [item for item in encoded_chunks if int(item["chain_index"]) == chain_index]
        chain_chunks = sorted(chain_chunks, key=lambda item: int(item["start"]))
        chain_embedding = torch.cat([item["embedding"] for item in chain_chunks], dim=0)
        if chain_embedding.shape[0] != chain_lengths[chain_index]:
            raise ValueError(
                f"Chain length mismatch for {row.pdb_id} chain {chain_index}: "
                f"expected {chain_lengths[chain_index]}, got {chain_embedding.shape[0]}"
            )
        chain_embeddings.append(chain_embedding)
        for item in chain_chunks:
            chunk_metadata.append(
                {
                    "chain_index": int(item["chain_index"]),
                    "chunk_index": int(item["chunk_index"]),
                    "start": int(item["start"]),
                    "end": int(item["end"]),
                }
            )

    embedding = torch.cat(chain_embeddings, dim=0)
    expected_total = int(row.protein_sequence_length)
    if embedding.shape[0] != expected_total:
        raise ValueError(f"Total length mismatch for {row.pdb_id}: expected {expected_total}, got {embedding.shape[0]}")

    return {
        "pdb_id": row.pdb_id,
        "split": row.split,
        "model_name": args.model_name,
        "sequence": row.protein_sequence,
        "sequence_length": expected_total,
        "chain_lengths": chain_lengths,
        "chunk_metadata": chunk_metadata,
        "embedding": embedding.to(torch.float16 if args.float16_output else torch.float32),
        "embedding_shape": list(embedding.shape),
        "pAffinity": float(row.pAffinity),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--limit", type=int, default=-1, help="Limit rows for local validation. Use -1 for all rows.")
    parser.add_argument("--row-batch-size", type=int, default=1, help="Reserved for future use; rows are cached one by one.")
    parser.add_argument("--chunk-batch-size", type=int, default=4)
    parser.add_argument("--max-residues-per-chunk", type=int, default=DEFAULT_MAX_RESIDUES_PER_CHUNK)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--float16-output", action="store_true", help="Store embeddings as float16 to reduce disk usage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    AutoTokenizer, EsmModel = load_transformers()
    device = select_device(args.device)

    manifest_path = project_path(args.manifest)
    cache_dir = project_path(args.cache_dir)
    report_json = project_path(args.report_json)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    if args.limit >= 0:
        df = df.head(args.limit).copy()

    print(f"Loading ESM tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = EsmModel.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model.eval()
    model.to(device)
    print(f"Loaded ESM model on {device}. Caching {len(df)} rows.")

    written = 0
    skipped_existing = 0
    failures = []
    embedding_shapes = []

    rows_iter = progress_bar(df.itertuples(index=False), desc="Cache ESM embeddings", unit="protein")
    for row in rows_iter:
        output_path = cache_dir / f"{row.pdb_id}.pt"
        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            try:
                cached = torch.load(output_path, map_location="cpu", weights_only=False)
                embedding_shapes.append(cached.get("embedding_shape", list(cached["embedding"].shape)))
            except Exception:
                pass
            continue
        try:
            cached = cache_one_row(row, tokenizer, model, device, args)
            torch.save(cached, output_path)
            embedding_shapes.append(cached["embedding_shape"])
            written += 1
        except Exception as exc:
            failures.append({"pdb_id": row.pdb_id, "split": row.split, "error": str(exc)})

    hidden_dims = sorted({int(shape[1]) for shape in embedding_shapes if len(shape) == 2})
    lengths = [int(shape[0]) for shape in embedding_shapes if len(shape) == 2]
    report = {
        "manifest": display_path(manifest_path),
        "cache_dir": display_path(cache_dir),
        "model_name": args.model_name,
        "device": str(device),
        "input_rows": int(len(df)),
        "written": written,
        "skipped_existing": skipped_existing,
        "failure_count": len(failures),
        "failures": failures[:20],
        "hidden_dims": hidden_dims,
        "sequence_length_min": min(lengths) if lengths else None,
        "sequence_length_max": max(lengths) if lengths else None,
        "sequence_length_mean": float(sum(lengths) / len(lengths)) if lengths else None,
        "max_residues_per_chunk": args.max_residues_per_chunk,
        "chunk_batch_size": args.chunk_batch_size,
        "float16_output": bool(args.float16_output),
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Cached ESM embeddings in: {display_path(cache_dir)}")
    print(f"Wrote report: {display_path(report_json)}")
    print(f"Rows: {len(df)}, written: {written}, skipped_existing: {skipped_existing}, failures: {len(failures)}")
    print(f"Hidden dims: {hidden_dims}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
