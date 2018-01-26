#
# import enumerations
#

from .._property import DatasetFillValueStatus
from .._property import DatasetFillTime
from .._property import DatasetAllocTime
from .._property import DatasetLayout
from .._property import LibVersion
from .._property import CopyFlag

def CopyFlag_or(self,b):
    if isinstance(b,(CopyFlag,CopyFlags)):
        return CopyFlags(self) | b
    else:
        raise TypeError("RHS of | operator must be a CopyFlag instance!")
    
CopyFlag.__or__ = CopyFlag_or

#
# import utility classes
#
from .._property import CopyFlags
from .._property import ChunkCacheParameters
from .._property import CreationOrder

#
# import propety list classes
#
from .._property import List
from .._property import DatasetTransferList
from .._property import FileAccessList
from .._property import FileMountList
from .._property import LinkAccessList
from .._property import ObjectCopyList
from .._property import ObjectCreationList
from .._property import StringCreationList
from .._property import DatasetAccessList
from .._property import DatatypeAccessList
from .._property import GroupAccessList
from .._property import DatasetCreationList
from .._property import GroupCreationList
from .._property import TypeCreationList
from .._property import AttributeCreationList
from .._property import LinkCreationList