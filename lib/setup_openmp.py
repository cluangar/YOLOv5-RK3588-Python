import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
setup(
    cmdclass = {"build_ext": build_ext},
    ext_modules = [
        Extension(
            "cython_post",
            ["yolov5_post_process_fast.pyx"],
            extra_compile_args = ["-fopenmp"],
            extra_link_args = ["-fopenmp"],
        include_dirs=[numpy.get_include()]
        )
    ]
)
