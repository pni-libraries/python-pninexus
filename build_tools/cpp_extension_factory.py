from .build_configuration import BuildConfiguration
from setuptools import Extension


class CppExtensionFactory(object):
    """

    :param BuildConfiguration config: build configuration
    """

    def __init__(self, config=None):

        if not isinstance(config, BuildConfiguration):
            raise TypeError(
                "The config argument must be "
                "an instance of BuildConfiguration!")

        self._config = config

    def create(self, module_name, source_files):

        return Extension(module_name, source_files,
                         include_dirs=self._config.include_directories,
                         library_dirs=self._config.library_directories,
                         libraries=self._config.link_libraries,
                         extra_link_args=self._config.linker_arguments,
                         language="c++",
                         extra_compile_args=self._config.compiler_arguments)
