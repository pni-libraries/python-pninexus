try:
    from subprocess import check_output
    
    def execute(l):
        return check_output(l)

except:
    from subprocess import Popen
    from subprocess import PIPE

    def execute(l):
        p = Popen(l,stdout=PIPE)
        result = ""

        for x in p.stdout: result+=x

        return result


def strip_string_list(inlist):
    """
    strip_string_list(inlist):
    Strip all strings in a list of strings from all leading and trailing blanks.

    input arguments:
    inlist ............ input list of strings

    return:
    new list with all strings stripped.
    """
    l = []
    for value in inlist:
        l.append(value.strip())

    return l

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
    for i in range(cnt): outlist.remove('')

    return outlist

def split_result(result,key):
    result = result.strip()
    result = result.split(key)
    result = remove_empty_strings(result)
    return result



class package(object):
    command = 'pkg-config'
    def __init__(self,pkgname):
        self.name = pkgname

    def _get_library_dirs(self):
        result = execute([self.command,'--libs-only-L',self.name])
        result = split_result(result,'-L')

        return result

    def _get_include_dirs(self):
        result = execute([self.command,'--cflags-only-I',self.name])
        return split_result(result,'-I')

    def _get_libraries(self):
        result = execute([self.command,'--libs-only-l',self.name])
        return split_result(result,'-l')

    def _get_compiler_flags(self):
        #first we obtain all compiler flags
        total_result = execute([self.command,'--cflags',self.name])
        total_result = total_result.strip()
        total_result = total_result.split(" ")
        total_result = remove_empty_strings(total_result)

        #now we have to obtain all the include files
        includes = execute([self.command,'--cflags-only-I',self.name])
        includes = includes.strip()
        includes = includes.split(" ")
        includes = remove_empty_strings(includes)

        for header in includes:
            total_result.remove(header)

        return total_result


    library_dirs = property(_get_library_dirs)
    libraries    = property(_get_libraries)
    compiler_flags = property(_get_compiler_flags)
    include_dirs = property(_get_include_dirs)

#testing routine
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print "You have to pass a package name to as a command line argument!"
        sys.exit()

    name = sys.argv[1]

    p = package(name)
    print "library directories: ",p.library_dirs
    print "libraries          : ",p.libraries
    print "compiler flags     : ",p.compiler_flags
    print "include directories: ",p.include_dirs

