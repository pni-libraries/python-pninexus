#
# import enumerations
#

from pni.io.h5cpp._property import DatasetFillValueStatus
from pni.io.h5cpp._property import DatasetFillTime
from pni.io.h5cpp._property import DatasetAllocTime
from pni.io.h5cpp._property import DatasetLayout
from pni.io.h5cpp._property import LibVersion
from pni.io.h5cpp._property import CopyFlag

def CopyFlag_or(self,b):
    if isinstance(b,(CopyFlag,CopyFlags)):
        return CopyFlags(self) | b
    else:
        raise TypeError("RHS of | operator must be a CopyFlag instance!")
    
CopyFlag.__or__ = CopyFlag_or

#
# import utility classes
#
from pni.io.h5cpp._property import CopyFlags
from pni.io.h5cpp._property import ChunkCacheParameters
from pni.io.h5cpp._property import CreationOrder

#
# import propety list classes
#
from pni.io.h5cpp._property import List
from pni.io.h5cpp._property import DatasetTransferList
from pni.io.h5cpp._property import FileAccessList
from pni.io.h5cpp._property import FileCreationList
from pni.io.h5cpp._property import FileMountList
from pni.io.h5cpp._property import LinkAccessList
from pni.io.h5cpp._property import ObjectCopyList
from pni.io.h5cpp._property import ObjectCreationList
from pni.io.h5cpp._property import StringCreationList
from pni.io.h5cpp._property import DatasetAccessList
from pni.io.h5cpp._property import DatatypeAccessList
from pni.io.h5cpp._property import GroupAccessList
from pni.io.h5cpp._property import DatasetCreationList
from pni.io.h5cpp._property import GroupCreationList
from pni.io.h5cpp._property import TypeCreationList
from pni.io.h5cpp._property import AttributeCreationList
from pni.io.h5cpp._property import LinkCreationList