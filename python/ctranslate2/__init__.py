import sys

if sys.platform == "win32":
    import ctypes
    import glob
    import os

    from importlib.resources import files

    module_name = sys.modules[__name__].__name__
    package_dir = str(files(module_name))

    # The libraries package name differs across ROCm releases.
    site_dir = os.path.dirname(package_dir)
    dll_dirs = [os.path.join(site_dir, "_rocm_sdk_core", "bin")]
    dll_dirs += sorted(glob.glob(os.path.join(site_dir, "_rocm_sdk_libraries*", "bin")))
    dll_dirs = [path for path in dll_dirs if os.path.isdir(path)]
    # Fallback only: the search order between DLL directories is unspecified.
    if not dll_dirs and os.environ.get("HIP_PATH"):
        dll_dirs = [os.path.join(os.environ["HIP_PATH"], "bin")]

    for path in [package_dir] + dll_dirs:
        try:
            os.add_dll_directory(path)
        except OSError:
            pass

    # Builds with WITH_CUDNN=ON (off by default) also link cudnn64_9.dll, which is not bundled.
    # Its folder must then be added here with os.add_dll_directory, e.g. CUDA_PATH\bin, or
    # CUDA_PATH\bin\x64 with CUDA 13 (the CUDA 12 and 13 builds of cuDNN share the file name).

    for library in glob.glob(os.path.join(package_dir, "*.dll")):
        ctypes.CDLL(library)

try:
    from ctranslate2._ext import (
        AsyncGenerationResult,
        AsyncScoringResult,
        AsyncTranslationResult,
        DataType,
        Device,
        Encoder,
        EncoderForwardOutput,
        ExecutionStats,
        GenerationResult,
        GenerationStepResult,
        Generator,
        MpiInfo,
        ScoringResult,
        StorageView,
        TranslationResult,
        Translator,
        contains_model,
        get_cuda_device_count,
        get_supported_compute_types,
        set_random_seed,
    )
    from ctranslate2.extensions import register_extensions
    from ctranslate2.logging import get_log_level, set_log_level

    register_extensions()
    del register_extensions
except ImportError as e:
    # Allow using the Python package without the compiled extension.
    if "No module named" in str(e):
        pass
    else:
        raise

from ctranslate2 import models
from ctranslate2.version import __version__


def _load_cuda_libraries():
    # The "cuda" extra installs cuBLAS and NVRTC from NVIDIA's pip packages, which are not on
    # the library search path. Load them by path so that the CUDA backend finds them by name,
    # unless a library with the same name is already loaded (e.g. by PyTorch). On Windows,
    # PyTorch's CUDA builds bundle the same libraries: those come first, so that PyTorch keeps
    # its own copies when it is imported later.
    import ctypes
    import glob
    import os
    import platform

    from importlib.util import find_spec

    def package_dirs(name):
        try:
            spec = find_spec(name)
        except (ImportError, ValueError):
            return []
        if spec is None or spec.submodule_search_locations is None:
            return []
        return list(spec.submodule_search_locations)

    # The CUDA 12 wheels have a local version label such as +cu128 (see setup.py).
    local_label = __version__.partition("+")[2]
    cuda12 = any(part.startswith("cu12") for part in local_label.split("."))

    if sys.platform == "win32" and platform.machine() == "AMD64":
        if cuda12:
            nvidia_dirs = [("cublas", "bin"), ("cuda_nvrtc", "bin")]
            names = [
                "cublasLt64_12.dll",
                "cublas64_12.dll",
                "nvrtc-builtins64_12*.dll",
                "nvrtc64_120_0.dll",
            ]
        else:
            nvidia_dirs = [("cu13", "bin", "x86_64")]
            names = [
                "cublasLt64_13.dll",
                "cublas64_13.dll",
                "nvrtc-builtins64_13*.dll",
                "nvrtc64_130_0.dll",
            ]
        library_dirs = [os.path.join(path, "lib") for path in package_dirs("torch")]
        library_dirs += [
            os.path.join(path, *subdir)
            for path in package_dirs("nvidia")
            for subdir in nvidia_dirs
        ]
        get_module_handle = ctypes.WinDLL("kernel32").GetModuleHandleW
        get_module_handle.argtypes = [ctypes.c_wchar_p]
        get_module_handle.restype = ctypes.c_void_p

        def is_loaded(name):
            return get_module_handle(name) is not None

    elif sys.platform == "linux" and platform.machine() == "x86_64":
        if cuda12:
            nvidia_dirs = [("cublas", "lib"), ("cuda_nvrtc", "lib")]
            names = [
                "libcublasLt.so.12",
                "libcublas.so.12",
                "libnvrtc-builtins.so.12*",
                "libnvrtc.so.12",
            ]
        else:
            nvidia_dirs = [("cu13", "lib")]
            names = [
                "libcublasLt.so.13",
                "libcublas.so.13",
                "libnvrtc-builtins.so.13*",
                "libnvrtc.so.13",
            ]
        library_dirs = [
            os.path.join(path, *subdir)
            for path in package_dirs("nvidia")
            for subdir in nvidia_dirs
        ]

        def is_loaded(name):
            try:
                ctypes.CDLL(name, mode=os.RTLD_NOLOAD)
                return True
            except OSError:
                return False

    else:
        return

    for name in names:
        paths = [
            path for d in library_dirs for path in glob.glob(os.path.join(d, name))
        ]
        for path in paths:
            if is_loaded(os.path.basename(path)):
                break
            try:
                ctypes.CDLL(path)
                break
            except OSError:
                pass


_load_cuda_libraries()
del _load_cuda_libraries

# converters and specs import torch (and, for converters, transformers) at module level.
# Those dependencies are only needed to convert models, not to run inference, so import
# these submodules on first use to keep "import ctranslate2" free of them.
_LAZY_SUBMODULES = ("converters", "specs")


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))


# A wildcard import resolves ``__all__`` when it is defined and the module globals
# otherwise, so without this the lazy submodules would silently drop out of
# ``from ctranslate2 import *``. Deriving the list keeps the wildcard surface identical
# to what it was before they became lazy; a wildcard import asks for everything, so
# resolving them here is expected.
__all__ = sorted(
    [name for name in globals() if not name.startswith("_")] + list(_LAZY_SUBMODULES)
)
