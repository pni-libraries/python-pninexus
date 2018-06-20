from .build_configuration import BuildConfiguration
from .ConanConfig import ConanBuildInfo


class ConanBuildInfoBuilder(object):
    """Construct a BuildConfiguration instance from conan build information

    """

    def create(self, filename):

        conan_config = ConanBuildInfo(filename)
        config = BuildConfiguration()

        config.include_directories = conan_config.includedirs
        config.library_directories = conan_config.libdirs
        config.link_libraries = conan_config.libs

        for libdir in conan_config.libdirs:
            config.add_linker_argument('-Wl,-rpath,' + libdir)

        return config
