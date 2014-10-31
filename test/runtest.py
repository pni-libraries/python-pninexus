#!/usr/bin/env python

import unittest
import nxfile_test
import nxgroup_test
#import NXFieldTest
#import nxfield_common_test

suite = unittest.TestSuite()
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(nxfile_test))
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(nxgroup_test))
#suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(nxfield_common_test))
runner = unittest.TextTestRunner()
result = runner.run(suite)


#=================running test for fields==================================
#
#for t in  NXFieldTest.types:
#    print "Running unit test for nxfield with type ",t
#    suite = unittest.TestSuite()
#    NXFieldTest.nxfield_test._typecode = t
#    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(NXFieldTest))
#
#    runner = unittest.TextTestRunner()
#    result = runner.run(suite)


#===========================running regression tests==========================
print "====================Regressions tests================================="
#import issue_53_test
#import issue_48_test
#reg_suite = unittest.TestSuite()
#reg_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(issue_53_test))
#reg_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(issue_48_test))
#
#reg_runner = unittest.TextTestRunner()
#reg_result = reg_runner.run(reg_suite)



