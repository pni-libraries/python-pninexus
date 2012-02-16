#!/usr/bin/env python

import unittest
import NXFileTest

suite = unittest.TestSuite()
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(NXFileTest))

runner = unittest.TextTestRunner()
result = runner.run(suite)

