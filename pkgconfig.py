
from __future__ import print_function
import sys


try:
    from subprocess import check_output

    def execute(lt):
        return check_output(lt)

except Exception:
    from subprocess import Popen
    from subprocess import PIPE

    def execute(lt):
        p = Popen(lt, stdout=PIPE)
        result = ""

        for x in p.stdout:
            result += x

        return result


def strip_string_list(inlist):
    """
    strip_string_list(inlist):
    Strip all strings in a list of strings from all leading and
    trailing blanks.

    input arguments:
    inlist ............ input list of strings

    return:
    new list with all strings stripped.
    """
    lt = []
    for value in inlist:
        lt.append(value.strip())

    return lt


def remove_empty_strings(inlist):
    """
    remove_empty_strings(inlist):
    Remove all empty strings from the list of strings.

    input arguments:
    inlist ............. inpust list of strings

    return:
    list without empty strings
    """

    cnt = inlist.count('')
    outlist = list(inlist)
    for i in range(cnt):
        outlist.remove('')

    return outlist


def split_result(result, key):
    result = result.strip()
    result = result.split(key)
    result = remove_empty_strings(result)
    return result


class package(object):
    command = 'pkg-config'

    def __init__(self, pkgname):
        self.name = pkgname

    def _decode(self, data):
        if sys.version_info.major >= 3:
            return data.decode('utf-8')
        else:
            return data

    def _get_library_dirs(self):
        result = self._decode(
            execute([self.command, '--libs-only-L', self.name]))
        result = split_result(result, '-L')
        return strip_string_list(result)

    def _get_include_dirs(self):
        result = self._decode(
            execute([self.command, '--cflags-only-I', self.name]))
        result = split_result(result, '-I')
        return strip_string_list(result)

    def _get_libraries(self):
        result = self._decode(
            execute([self.command, '--libs-only-l', self.name]))
        result = split_result(result, '-l')
        return strip_string_list(result)

    def _get_compiler_flags(self):
        # first we obtain all compiler flags
        total_result = self._decode(
            execute([self.command, '--cflags', self.name]))
        total_result = total_result.strip()
        total_result = total_result.split(" ")
        total_result = remove_empty_strings(total_result)

        # now we have to obtain all the include files
        includes = self._decode(
            execute([self.command, '--cflags-only-I', self.name]))
        includes = includes.strip()
        includes = includes.split(" ")
        includes = remove_empty_strings(includes)

        for header in includes:
            total_result.remove(header)

        return total_result

    library_dirs = property(_get_library_dirs)
    libraries = property(_get_libraries)
    compiler_flags = property(_get_compiler_flags)
    include_dirs = property(_get_include_dirs)


# testing routine
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("You have to pass a package name to as a command line argument!")
        sys.exit()

    name = sys.argv[1]

    p = package(name)
    print("library directories: ", p.library_dirs)
    print("libraries          : ", p.libraries)
    print("compiler flags     : ", p.compiler_flags)
    print("include directories: ", p.include_dirs)
