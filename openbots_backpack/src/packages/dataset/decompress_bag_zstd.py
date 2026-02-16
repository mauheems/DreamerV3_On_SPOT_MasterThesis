#!/usr/bin/env python3
"""
Decompress a rosbag2 zstd-compressed .db3.zstd file to .db3
and update metadata.yaml to mark the bag as uncompressed.
"""

import argparse
from pathlib import Path
import sys

try:
    import zstandard as zstd
except ImportError as exc:
    raise SystemExit("Missing dependency: zstandard. Install with: python3 -m pip install zstandard") from exc

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: PyYAML. Install with: python3 -m pip install pyyaml") from exc


def decompress_bag(bag_dir: Path) -> Path:
    if not bag_dir.is_dir():
        raise FileNotFoundError(f"Bag directory not found: {bag_dir}")

    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found in {bag_dir}")

    with metadata_path.open("r") as f:
        meta = yaml.safe_load(f)

    info = meta.get("rosbag2_bagfile_information", {})
    rel_paths = info.get("relative_file_paths", [])
    if not rel_paths:
        raise ValueError("metadata.yaml has no relative_file_paths")

    # Assume single-file bags
    comp_name = rel_paths[0]
    comp_path = bag_dir / comp_name

    if comp_path.suffix != ".zstd":
        raise ValueError(f"Expected .zstd file, got: {comp_path.name}")

    out_path = comp_path.with_suffix("")  # remove .zstd

    if not out_path.exists():
        dctx = zstd.ZstdDecompressor()
        with comp_path.open("rb") as f_in, out_path.open("wb") as f_out:
            dctx.copy_stream(f_in, f_out)
    else:
        print(f"Uncompressed file already exists: {out_path}")

    # Update metadata to uncompressed
    info["compression_format"] = ""
    info["compression_mode"] = "NONE"
    info["relative_file_paths"] = [out_path.name]
    if info.get("files"):
        info["files"][0]["path"] = out_path.name

    with metadata_path.open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Decompress rosbag2 .db3.zstd and update metadata.yaml")
    parser.add_argument("bag_dir", help="Path to rosbag directory (contains metadata.yaml)")
    args = parser.parse_args()

    out_path = decompress_bag(Path(args.bag_dir))
    print(f"Decompressed bag to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
