#!/usr/bin/env python

import unittest
import NXFileTest
import NXGroupTest
import NXFieldTest

suite = unittest.TestSuite()
#suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(NXFileTest))
#suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(NXGroupTest))
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(NXFieldTest))

runner = unittest.TextTestRunner()
result = runner.run(suite)

