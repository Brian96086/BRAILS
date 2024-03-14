from setuptools import setup,find_packages
from os import path as os_path

import brails
import glob
import os
import subprocess

#Segmentation Model requires torch beforehand during package installation
def install_torch():
     try:
         import torch
     except ImportError:
         subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

 # Call the function to ensure torch is installed
install_torch()

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

this_directory = os_path.abspath(os_path.dirname(__file__))


def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


#Restructured for segmentation models(GroundedSAM)
def read_requirements(fname, with_version = True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                info["package"] = line
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest  # NOQA
                    info["version"] = (op, version)
            yield info
            
    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages

    # return [line.strip() for line in read_file(fname).splitlines()
    #         if not line.startswith('#')]

#for segmentation (GroundedSAM)
def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "groundingdino", "models", "GroundingDINO", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources

    # We need these variables to build with CUDA when we create the Docker image
    # It solves https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/53
    # and https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/84 when running
    # inside a Docker container.
    am_i_docker = os.environ.get('AM_I_DOCKER', '').casefold() in ['true', '1', 't']
    use_cuda = os.environ.get('BUILD_WITH_CUDA', '').casefold() in ['true', '1', 't']

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or \
            (am_i_docker and use_cuda):
        print("Compiling with CUDA")
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        print("Compiling without CUDA")
        define_macros += [("WITH_HIP", None)]
        extra_compile_args["nvcc"] = []
        return None

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "groundingdino._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name='BRAILS',
    python_requires='>=3.6',
    version=brails.__version__,
    description="Building and Infrastructure Recognition Using AI at Large-Scale",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    author="NHERI SimCenter",
    author_email='nheri-simcenter@berkeley.edu',
    url='https://github.com/NHERI-SimCenter/BRAILS',
    #packages=['brails'],
    packages=find_packages(),
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    zip_safe=False,
    install_requires=read_requirements('requirements.txt'),
    include_package_data=True,
    package_data={'': ['brails/modules/FoundationClassifier/csail_segmentation_tool/csail_seg/data/color150.mat']
                  },
    license="BSD 3-Clause",
    keywords=['brails', 'bim', 'brails framework'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Natural Language :: English',
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
