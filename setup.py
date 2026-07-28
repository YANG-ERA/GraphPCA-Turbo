import io
import os
import sys
import warnings
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

here = os.path.abspath(os.path.dirname(__file__))

# Package meta-data.
NAME = 'st-graphpca'
DESCRIPTION = 'Scalable hierarchical graph-regularized dimension reduction for single- and multi-sample spatial transcriptomics.'
EMAIL = '599568651@qq.com'
URL = "https://github.com/YANG-ERA/GraphPCA-Turbo"
AUTHOR = 'Jiyuan Yang'
VERSION = '2.1.0'

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# ==========================================
# C++ Extension Compilation Configurations
# ==========================================
def find_eigen():
    """Automatically find the Eigen3 include directory in common paths."""
    search_paths = [
        os.path.join(sys.prefix, 'include', 'eigen3'),  # Conda standard path
        os.path.join(sys.prefix, 'include'),            # Conda fallback
        '/opt/homebrew/include/eigen3',                 # Mac Homebrew
        '/usr/local/include/eigen3',                    # Linux/Mac default
        '/usr/include/eigen3',                          # Linux standard
        os.environ.get('EIGEN3_INCLUDE_DIR', '')        # Allow manual override via env var
    ]
    
    for path in search_paths:
        if path and os.path.exists(os.path.join(path, 'Eigen', 'Core')):
            print(f"✅ Found Eigen3 at: {path}")
            return path
            
    # CRITICAL FIX: Return None instead of raising an error to allow graceful fallback
    return None

ext_modules = []

# Try to configure C++ extension, fallback to pure Python if dependencies are missing
try:
    import pybind11
    eigen_include_dir = find_eigen()
    
    if eigen_include_dir:
        ext_modules = [
            Extension(
                'gpca_cpp',  
                ['gpca_core.cpp'], 
                include_dirs=[
                    pybind11.get_include(),
                    pybind11.get_include(user=True),
                    eigen_include_dir, 
                ],
                language='c++'
            ),
        ]
    else:
        warnings.warn("\n⚠️ WARNING: Eigen3 not found. Skipping C++ extension build. The pure Python version will be built.\n")
except ImportError:
    warnings.warn("\n⚠️ WARNING: pybind11 not found. Skipping C++ extension build. The pure Python version will be built.\n")

class BuildExt(build_ext):
    def build_extensions(self):
        opts = ['-O3', '-Wall', '-shared', '-std=c++14', '-fPIC']
        
        # Optimization for macOS clang compiler
        if sys.platform == 'darwin':
            opts.append('-stdlib=libc++')
            
        for ext in self.extensions:
            ext.extra_compile_args = opts
        super().build_extensions()

# ==========================================
# Main Setup Module
# ==========================================
setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    license='MIT',
    description=DESCRIPTION,
    url=URL,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    install_requires=[
        "numpy>=1.23.4",
        "pandas>=2.1.0",
        "matplotlib>=3.7.3",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.1",
        "networkx>=3.2.1",
        "scanpy>=1.9.8",
        "squidpy>=1.4.1",
        "pybind11>=2.10.0"
    ]
)
