from pninexus.h5cpp._filter import Filter
from pninexus.h5cpp._filter import Deflate
from pninexus.h5cpp._filter import Shuffle
from pninexus.h5cpp._filter import Fletcher32
from pninexus.h5cpp._filter import NBit
from pninexus.h5cpp._filter import SZip
from pninexus.h5cpp._filter import SZipOptionMask
from pninexus.h5cpp._filter import ScaleOffset
from pninexus.h5cpp._filter import SOScaleType
from pninexus.h5cpp._filter import ExternalFilter
from pninexus.h5cpp._filter import Availability
from pninexus.h5cpp._filter import is_filter_available
from pninexus.h5cpp._filter import _externalfilters_fill


class ExternalFilters(list):
    """ List of ExternalFilters"""

    def fill(self, dcpl, max_cd_number=16, max_name_size=257):
        """ fills the list with conntent provided by  DatasetCreationList

        :param dcpl: dataset creationlist
        :type dcpl: :class:`DatasetCreationList`
        :param max_cd_number: maximal cd parameters number
        :type max_cd_number: int
        :param max_name_size: maximal name size
        :type max_name_size: int
        :returns: a list of Availability flags
        :rtype: :obj: `list` <:class:`Availability`>
        """
        return _externalfilters_fill(self, dcpl, max_cd_number, max_name_size)


__all__ = ["Filter", "Deflate", "Shuffle", "Fletcher32", "NBit", "SZip",
           "SZippOptionMask"
           "ExternalFilter", "ExternalFilters", "Availability", "ScaleOffset",
           "SOScaleType", "is_filter_available"]
