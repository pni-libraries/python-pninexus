#
# (c) Copyright 2018 DESY,
#
# This file is part of python-pninexus.
#
# python-pninexus is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# python-pninexus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with python-pninexus.  If not, see <http://www.gnu.org/licenses/>.
# ===========================================================================
#
# Created on: Jan 26, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
import unittest

from pninexus.h5cpp.property import ChunkCacheParameters


class ChunkCacheParametersTest(unittest.TestCase):

    def testConstruction(self):

        params = ChunkCacheParameters(1000000, 20313, 9.234)
        self.assertEqual(params.chunk_slots, 1000000)
        self.assertEqual(params.chunk_cache_size, 20313)
        self.assertEqual(params.preemption_policy, 9.234)

    def testSetters(self):

        params = ChunkCacheParameters()
        params.chunk_slots = 3400000
        self.assertEqual(params.chunk_slots, 3400000)

        params.chunk_cache_size = 20313
        self.assertEqual(params.chunk_cache_size, 20313)

        params.preemption_policy = 934.234
        self.assertEqual(params.preemption_policy, 934.234)
