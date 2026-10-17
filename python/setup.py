import glob
import os
import sys

import pybind11

from pybind11.setup_helpers import ParallelCompile
from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py

base_dir = os.path.dirname(os.path.abspath(__file__))
include_dirs = [pybind11.get_include()]
library_dirs = []


def _get_long_description():
    readme_path = os.path.join(base_dir, "README.md")
    if not os.path.exists(readme_path):
        return ""
    with open(readme_path, encoding="utf-8") as readme_file:
        return readme_file.read()


def _get_project_version():
    version_path = os.path.join(base_dir, "ctranslate2", "version.py")
    version = {}
    with open(version_path, encoding="utf-8") as fp:
        exec(fp.read(), version)
    return version["__version__"]


def _maybe_add_library_root(lib_name):
    if "%s_ROOT" % lib_name in os.environ:
        root = os.environ["%s_ROOT" % lib_name]
        include_dirs.append("%s/include" % root)
        for lib_dir in ("lib", "lib64"):
            path = "%s/%s" % (root, lib_dir)
            if os.path.exists(path):
                library_dirs.append(path)
                break


_maybe_add_library_root("CTRANSLATE2")

cflags = ["-std=c++17", "-fvisibility=hidden"]
ldflags = []
package_data = {}
if sys.platform == "darwin":
    # std::visit requires macOS 10.14
    cflags.append("-mmacosx-version-min=10.14")
    ldflags.append("-Wl,-rpath,/usr/local/lib")
elif sys.platform == "win32":
    cflags = ["/std:c++17", "/d2FH4-"]
    package_data["ctranslate2"] = ["*.dll"]
elif sys.platform == "linux":
    cflags.append("-fPIC")
    ldflags.append("-Wl,-rpath,/usr/local/lib64:/usr/local/lib")

ctranslate2_module = Extension(
    "ctranslate2._ext",
    sources=glob.glob(os.path.join("cpp", "*.cc")),
    extra_compile_args=cflags,
    extra_link_args=ldflags,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["ctranslate2"],
)

ParallelCompile("CMAKE_BUILD_PARALLEL_LEVEL").install()

# The CUDA version the library is built with: 13.0 for the PyPI wheels, or 12.8 for the wheels
# attached to GitHub releases. CTRANSLATE2_CUDA_VERSION sets it for this script and for
# python/tools/prepare_build_environment_*.sh. NVIDIA's CUDA 13 packages group NVRTC with
# cuBLAS, which can load it at run time; for CUDA 12, its newest cuBLAS brings NVRTC along.
cuda_version = os.environ.get("CTRANSLATE2_CUDA_VERSION", "13.0")
cuda_requirements = {
    "13.0": ["nvidia-cublas>=13.1.1.3,<14", "nvidia-cuda-nvrtc>=13.0.88,<14"],
    "12.8": ["nvidia-cublas-cu12>=12.8.4.1,<13"],
}
if cuda_version not in cuda_requirements:
    raise ValueError(f"Unsupported CTRANSLATE2_CUDA_VERSION {cuda_version}")

# The platforms with CUDA wheels.
cuda_platforms = (
    '(sys_platform == "linux" and platform_machine == "x86_64")'
    ' or (sys_platform == "win32" and platform_machine == "AMD64")'
)

# Wheels for other CUDA versions get a local version label, e.g. 4.8.3.post2+cu128, which also
# tells the loader in __init__.py which cuBLAS to look for.
base_version = _get_project_version()
project_version = base_version
if cuda_version != "13.0":
    separator = "." if "+" in base_version else "+"
    project_version += separator + "cu" + cuda_version.replace(".", "")


class BuildPy(build_py):
    def run(self):
        super().run()
        # Give the package the labeled version too, so that __version__ matches the wheel.
        package_dir = os.path.join(self.build_lib, "ctranslate2")
        if project_version != base_version and os.path.isdir(package_dir):
            source_path = os.path.join(base_dir, "ctranslate2", "version.py")
            with open(source_path, encoding="utf-8") as version_file:
                text = version_file.read()
            text = text.replace(f'"{base_version}"', f'"{project_version}"')
            target_path = os.path.join(package_dir, "version.py")
            with open(target_path, "w", encoding="utf-8") as version_file:
                version_file.write(text)


setup(
    name="ctranslate2-reborn",
    version=project_version,
    license="MIT",
    description="Unofficial fork of CTranslate2, a fast inference engine for Transformer models",
    long_description=_get_long_description(),
    long_description_content_type="text/markdown",
    author="Blair Chintella",
    url="https://github.com/BBC-Esq/CTranslate2-Reborn",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        f"Environment :: GPU :: NVIDIA CUDA :: {cuda_version.split('.')[0]}",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Source": "https://github.com/BBC-Esq/CTranslate2-Reborn",
        "Issues": "https://github.com/BBC-Esq/CTranslate2-Reborn/issues",
    },
    keywords="nmt neural machine translation cuda mkl inference quantization",
    packages=find_packages(exclude=["bin"]),
    package_data=package_data,
    ext_modules=[ctranslate2_module],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pyyaml>=5.3,<7",
    ],
    extras_require={
        "cuda": [
            f"{requirement}; {cuda_platforms}"
            for requirement in cuda_requirements[cuda_version]
        ],
    },
    cmdclass={"build_py": BuildPy},
    entry_points={
        "console_scripts": [
            "ct2-fairseq-converter=ctranslate2.converters.fairseq:main",
            "ct2-marian-converter=ctranslate2.converters.marian:main",
            "ct2-openai-gpt2-converter=ctranslate2.converters.openai_gpt2:main",
            "ct2-opennmt-py-converter=ctranslate2.converters.opennmt_py:main",
            "ct2-opennmt-tf-converter=ctranslate2.converters.opennmt_tf:main",
            "ct2-opus-mt-converter=ctranslate2.converters.opus_mt:main",
            "ct2-transformers-converter=ctranslate2.converters.transformers:main",
        ],
    },
)
