"""Helpers for loading models and datasets from ModelScope."""
import os
from pathlib import Path


# Try these ModelScope mirrors when the short experiment alias is requested.
_MODEL_ALIASES = {
    "llama-7b": (
        # Verified ModelScope mirrors; shakechen avoids downloading duplicate
        # pickle weights by requesting only its safetensors files below.
        "shakechen/Llama-2-7b-hf",
        "vllm-ascend/Llama-2-7b-hf",
        # Keep older cache IDs discoverable even though their remote mirrors 404.
        "LLM-Research/llama-2-7b-hf",
        "AI-ModelScope/llama-2-7b-hf",
    )
}


_DATASET_IDS = {
    "wikitext": "modelscope/wikitext",
    "ptb_text_only": "torushy/flm-ptb-text-only",
    "allenai/c4": "allenai/c4",
    "ai2_arc": "modelscope/ai2_arc",
    "glue": "modelscope/glue",
    "super_glue": "modelscope/super_glue",
    "piqa": "opencompass/piqa",
}


def modelscope_model_path(model, cache_dir=None, force_download=False):
    """Return a local model path, preferring local files over ModelScope."""
    if Path(model).exists():
        return model

    candidates = _MODEL_ALIASES.get(model.lower(), (model,))
    if not force_download:
        for candidate in candidates:
            cached = _find_modelscope_model_snapshot(candidate, cache_dir)
            if cached is not None:
                print(f"Using cached ModelScope model snapshot: {cached}")
                return str(cached)

    from modelscope import snapshot_download

    errors = []
    for candidate in candidates:
        try:
            allow_patterns = None
            if candidate == "shakechen/Llama-2-7b-hf":
                allow_patterns = [
                    "*.safetensors", "*.json", "tokenizer.model",
                    "LICENSE.txt", "USE_POLICY.md",
                ]
            kwargs = {"cache_dir": cache_dir}
            if allow_patterns is not None:
                kwargs["allow_patterns"] = allow_patterns
            return snapshot_download(candidate, **kwargs)
        except Exception as error:
            errors.append(f"{candidate}: {error}")
    raise RuntimeError(
        "Could not load a ModelScope model; tried:\n" + "\n".join(errors)
    )


def _modelscope_cache_roots(cache_dir=None):
    """Return plausible ModelScope cache roots, most specific first."""
    paths = []
    env_roots = [
        cache_dir,
        os.environ.get("MODELSCOPE_CACHE"),
        os.environ.get("MODELSCOPE_CACHE_HOME"),
    ]
    try:
        env_roots.append(Path.home() / ".cache" / "modelscope")
    except RuntimeError:
        # Path.home() can fail in unusual container environments.
        pass

    for path in env_roots:
        if not path:
            continue
        root = Path(path).expanduser()
        # Depending on the ModelScope version, an explicit cache_dir is either
        # treated as the cache root or as the parent containing hub/datasets.
        paths.extend((root, root / "hub"))
    return paths


def _find_modelscope_model_snapshot(model, cache_dir=None):
    """Find a usable local ModelScope model snapshot without network I/O."""
    model_path = Path(model)

    def has_weights(snapshot):
        if not (snapshot / "config.json").is_file():
            return False

        def complete_index(index_name):
            index_path = snapshot / index_name
            if not index_path.is_file() or index_path.stat().st_size == 0:
                return False
            try:
                import json
                index = json.loads(index_path.read_text(encoding="utf-8"))
                required = set(index.get("weight_map", {}).values())
            except (OSError, ValueError, TypeError):
                return False
            return bool(required) and all(
                (snapshot / name).is_file() and (snapshot / name).stat().st_size > 0
                for name in required
            )

        if complete_index("model.safetensors.index.json"):
            return True
        if complete_index("pytorch_model.bin.index.json"):
            return True

        candidates = [
            *snapshot.glob("*.safetensors"),
            *snapshot.glob("pytorch_model*.bin"),
            *snapshot.glob("*.pth"),
        ]
        return any(path.is_file() and path.stat().st_size > 0 for path in candidates)

    for cache_root in _modelscope_cache_roots(cache_dir):
        legacy_path = None
        if len(model_path.parts) == 2:
            owner, name = model_path.parts
            legacy_path = Path(f"{owner}___{name}")
        roots = [
            cache_root / "models" / model_path,
            cache_root / "models" / legacy_path if legacy_path else None,
            cache_root / model_path,
            cache_root / legacy_path if legacy_path else None,
        ]
        roots = [root for root in roots if root is not None]
        if cache_root.name == model_path.name:
            roots.append(cache_root)
        for model_root in roots:
            candidates = [path for path in model_root.iterdir() if path.is_dir()] if model_root.is_dir() else []
            for snapshot in candidates:
                if has_weights(snapshot):
                    return snapshot
    return None


def _find_modelscope_dataset_snapshot(dataset_id, relative_files, cache_dir=None):
    """Find a complete local ModelScope dataset snapshot without network I/O."""
    dataset_path = Path(dataset_id)
    searched_dataset_roots = []

    for cache_root in _modelscope_cache_roots(cache_dir):
        # Common ModelScope layouts:
        #   <cache>/hub/datasets/<owner>/<dataset>/<revision>/<file>
        #   <cache>/datasets/<owner>/<dataset>/<revision>/<file>
        #   <cache>/<owner>/<dataset>/<revision>/<file>
        roots = [
            cache_root / "datasets" / dataset_path,
            cache_root / dataset_path,
        ]
        # Handle an explicit cache_dir that already points at the dataset root.
        if cache_root.name == dataset_path.name:
            roots.append(cache_root)
        for dataset_root in roots:
            if dataset_root in searched_dataset_roots:
                continue
            searched_dataset_roots.append(dataset_root)
            for relative_file in relative_files:
                relative = Path(relative_file)
                candidates = list(dataset_root.glob(str(relative)))
                candidates.extend(dataset_root.glob(f"*/{relative}"))
                candidates.extend(dataset_root.glob(f"**/{relative}"))
                for candidate in candidates:
                    if not candidate.is_file() or candidate.stat().st_size == 0:
                        continue
                    snapshot_root = candidate
                    for _ in relative.parts:
                        snapshot_root = snapshot_root.parent
                    if all(
                        (snapshot_root / needed).is_file()
                        and (snapshot_root / needed).stat().st_size > 0
                        for needed in relative_files
                    ):
                        return snapshot_root
    return None


def load_modelscope_dataset(
    path,
    name=None,
    split=None,
    data_dir=None,
    data_files=None,
    cache_dir=None,
    download_mode=None,
):
    """Load a ModelScope dataset using the project's Hugging Face names."""
    dataset_id = _DATASET_IDS.get(path, path)

    if path == "allenai/c4" and data_files:
        # ModelScope 1.18's dataset loader ignores the single-shard data_files
        # selection and tries to prepare the complete 365M-example C4 corpus.
        # Download only the requested shards, then build a local JSON dataset.
        from datasets import load_dataset
        from modelscope.hub.snapshot_download import dataset_snapshot_download

        relative_files = {
            split: [file if isinstance(file, str) else file[0] for file in files]
            if isinstance(files, (list, tuple)) else [files]
            for split, files in data_files.items()
        }
        patterns = sorted({file for files in relative_files.values() for file in files})
        root = None
        if os.environ.get("GPTQ_FORCE_DATASET_DOWNLOAD", "0") != "1":
            root = _find_modelscope_dataset_snapshot(dataset_id, patterns, cache_dir)
        if root is not None:
            print(f"Using cached ModelScope dataset snapshot: {root}")
        else:
            root = Path(dataset_snapshot_download(
                dataset_id,
                cache_dir=cache_dir,
                allow_patterns=patterns,
            ))
        local_files = {
            split: [str(root / file) for file in files]
            for split, files in relative_files.items()
        }
        return load_dataset("json", data_files=local_files, split=split)

    if path == "ptb_text_only":
        # ModelScope's PTB mirror is a datasets.save_to_disk() export. MsDataset
        # 1.18 mistakes its state.json files for source records, so read the
        # downloaded Arrow dataset directly.
        from datasets import load_from_disk
        from modelscope.hub.snapshot_download import dataset_snapshot_download

        root = Path(dataset_snapshot_download(dataset_id, cache_dir=cache_dir))
        loaded = load_from_disk(str(root / "dataset"))
        return loaded[split] if split is not None else loaded

    from modelscope.msdatasets import MsDataset

    if path == "allenai/c4" and name == "allenai--c4":
        name = "en"

    kwargs = {
        "subset_name": name,
        "split": split,
        "data_dir": data_dir,
        "data_files": data_files,
        "cache_dir": cache_dir,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if download_mode is not None:
        # ModelScope 1.18 uses its own enum, not datasets.DownloadMode.
        from modelscope.utils.constant import DownloadMode as ModelScopeDownloadMode

        mode_name = getattr(download_mode, "name", download_mode)
        if mode_name == "REUSE_CACHE_IF_EXISTS":
            mode_name = "REUSE_DATASET_IF_EXISTS"
        try:
            kwargs["download_mode"] = ModelScopeDownloadMode[mode_name]
        except KeyError as error:
            raise ValueError(f"Unsupported ModelScope download mode: {mode_name}") from error

    try:
        return MsDataset.load(dataset_id, **kwargs)
    except Exception as first_error:
        # Some ModelScope mirrors omit the original config name.
        if not name:
            raise first_error
        kwargs.pop("subset_name", None)
        try:
            return MsDataset.load(dataset_id, **kwargs)
        except Exception:
            raise first_error
