#!/usr/bin/env python

import unittest
import nxfile_test
import nxgroup_test
import nxfield_test
import nxfield_common_test

class test_result(object):
    def __init__(self,tr,description="",name=""):
        self._result = tr
        self._desc   = description
        self._name   = name

    def _get_name(self):
        return self._name

    def _set_name(self,value):
        self._name = value

    def _get_description(self):
        return self._desc

    def _set_description(self,value):
        self._desc = value

    name = property(_get_name,_set_name)
    description = property(_get_description,_set_description)

    def __unicode__(self):
        o = u"Test :       {}\n".format(self.name)
        o+= u"Description: {}\n".format(self.description)
        o+= u"All passed:  {}\n".format(self._result.wasSuccessful())

        o+= 80*u"-"+'\n'
        return o

    def __str__(self):
        return self.__unicode__()


results = []
suite = unittest.TestSuite()
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(nxfile_test))
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(nxgroup_test))
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(nxfield_common_test))
runner = unittest.TextTestRunner()
results.append(test_result(runner.run(suite),
               name="Basic object test",
               description="Testing basic functionality of all objects"))

#=================running test for fields==================================
#
for t in  nxfield_test.types:
    suite = unittest.TestSuite()
    nxfield_test.nxfield_test._typecode = t
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(nxfield_test))

    runner = unittest.TextTestRunner()
    results.append(test_result(runner.run(suite),
                   name = "Field test for "+t,
                   description = "Testing nxfield IO functionality for data"+\
                                 "data type ("+t+")"
                                 ))


#===========================running regression tests==========================
print "====================Regressions tests================================="
import issue_53_test
import issue_48_test
reg_suite = unittest.TestSuite()
reg_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(issue_53_test))
reg_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(issue_48_test))

reg_runner = unittest.TextTestRunner()
reg_result = reg_runner.run(reg_suite)

for r in results:
    print r



