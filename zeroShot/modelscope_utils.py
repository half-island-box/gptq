"""ModelScope helpers for running zeroShot as either a package or script."""
try:
    from modelscope_utils import load_modelscope_dataset, modelscope_model_path
except ImportError:  # python zeroShot/main.py
    import importlib.util
    from pathlib import Path

    _helper_path = Path(__file__).resolve().parents[1] / "modelscope_utils.py"
    _spec = importlib.util.spec_from_file_location(
        "_gptq_modelscope_utils", _helper_path
    )
    _helper = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_helper)
    load_modelscope_dataset = _helper.load_modelscope_dataset
    modelscope_model_path = _helper.modelscope_model_path

__all__ = ["load_modelscope_dataset", "modelscope_model_path"]
