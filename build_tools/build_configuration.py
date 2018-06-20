
def extend_or_append(input_list, value):

    if isinstance(value, (list, tuple)):
        input_list.extend(value)
    else:
        input_list.append(value)

    return input_list


class BuildConfiguration(object):
    """Data class storing build configuration

    This class stores some basic build configuration for C or C++ extensions,
    including

    * paths to header files
    * paths to the librarys
    * libraries to link to an extension
    * additional arguments to the linker
    * additional arguments to the compiler

    """
    def __init__(self):

        self._include_directories = []
        self._library_directories = []
        self._link_libraries = []
        self._linker_arguments = []
        self._compiler_arguments = []

    #
    # handling include directories
    #

    @property
    def include_directories(self):
        return self._include_directories

    @include_directories.setter
    def include_directories(self, directories):
        self._include_directories = directories

    def add_include_directories(self, directories):
        self._include_directories.extend(directories)

    def add_include_directory(self, directory):
        self._include_directories.append(directory)

    #
    # setting library directories
    #

    @property
    def library_directories(self):
        return self._library_directories

    @library_directories.setter
    def library_directories(self, directories):
        self._library_directories = directories

    def add_library_directories(self, directories):
        self._library_directories.extend(directories)

    def add_library_directory(self, directory):
        self._library_directories.append(directory)

    #
    # handling link libraries
    #

    @property
    def link_libraries(self):
        return self._link_libraries

    @link_libraries.setter
    def link_libraries(self, value):
        self._link_libraries = extend_or_append(self._link_libraries, value)

    def add_link_library(self, library):
        """add a library to the existing ones

        """
        self._link_libraries.append(library)

    def add_link_libraries(self, libraries):
        """adding a list of libraries to the exsting ones

        """
        self._link_libraries.extend(libraries)

    #
    # handling linker arguments
    #

    @property
    def linker_arguments(self):
        return self._linker_arguments

    @linker_arguments.setter
    def linker_arguments(self, arguments):
        self._linker_arguments = arguments

    def add_linker_argument(self, argument):
        self._linker_arguments.append(argument)

    def add_linker_arguments(self, arguments):
        self._linker_arguments.extend(arguments)

    #
    # handling compiler arguments
    #

    @property
    def compiler_arguments(self):
        return self._compiler_arguments

    @compiler_arguments.setter
    def compiler_arguments(self, arguments):
        self._compiler_arguments = arguments

    def add_compiler_argument(self, argument):
        self._compiler_arguments.append(argument)

    def add_compiler_arguments(self, arguments):
        self._compiler_arguments.extend(arguments)
