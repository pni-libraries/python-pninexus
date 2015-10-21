import sys

from .issue_48_test import *
from .issue_53_test import *
from .issue_3_regression_test import *
from .issue_4_regression_test import *

from .issue_7_regression_test import *

if sys.version_info.major<3:
    from .issue_6_regression_test import *
