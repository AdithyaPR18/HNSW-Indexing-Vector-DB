from setuptools import setup, Extension
import pybind11
import sys
import os

extra_compile_args = ["-O3", "-std=c++17", "-ffast-math"]
extra_link_args = []

if sys.platform == "darwin":
    import platform as _platform
    machine = _platform.machine()
    # Only add -march=native when not doing a universal2 fat-binary build
    if os.environ.get("ARCHFLAGS", "") == "" and machine != "x86_64":
        extra_compile_args.append("-mcpu=apple-m1")
    extra_compile_args += ["-mmacosx-version-min=11.0"]
    extra_link_args += ["-mmacosx-version-min=11.0"]
else:
    extra_compile_args.append("-march=native")

hnsw_ext = Extension(
    "hnsw_index",
    sources=[
        "src/bindings/bindings.cpp",
        "src/hnsw/hnsw.cpp",
    ],
    include_dirs=[
        pybind11.get_include(),
        "src/hnsw",
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

setup(
    name="hnsw_index",
    version="0.1.0",
    ext_modules=[hnsw_ext],
)
