from __future__ import print_function
import re


class ConanBuildInfo(object):
    """Reader for the conan build

    """

    section_re = re.compile(r"^\[(?P<NAME>[A-Za-z_0-9]+)\]$")

    def __init__(self, filename):

        with open(filename, "r") as f:
            current_key = None
            for line in f:
                line = line.strip()
                match = self.section_re.match(line)
                if match:
                    current_key = match.group("NAME")
                    self.__dict__[current_key] = []
                else:
                    if current_key and line and line != 'hello':
                        self.__dict__[current_key].append(line)
