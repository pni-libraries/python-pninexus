#!/usr/bin/env python

import unittest
import NXFileTest
import NXGroupTest
import NXFieldTest
import issue_53_test

suite = unittest.TestSuite()
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(NXFileTest))
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(NXGroupTest))
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(NXFieldTest))
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(issue_53_test))

runner = unittest.TextTestRunner()
result = runner.run(suite)

