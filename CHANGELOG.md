# ChangeLog

## 3.5.0 - 2026-08-10
- add wrapper for file drivers ([#242](https://github.com/pni-libraries/python-pninexus/pull/242)
- technical release for libpninexus v3.5.0 (https://github.com/pni-libraries/libpninexus/issues/280) and libh5cpp v0.8.0 (https://github.com/ess-dmsc/h5cpp/issues/679)

## 3.4.0 - 2026-05-05
- update tests for hdf5 2.0 ([#235](https://github.com/pni-libraries/python-pninexus/pull/235), [#233](https://github.com/pni-libraries/python-pninexus/pull/233)
- add support for windows ([#231](https://github.com/pni-libraries/python-pninexus/pull/231))
- cleaning setting string empty ([#229](https://github.com/pni-libraries/python-pninexus/pull/229))
- define empty python string with PyCONSTANT_EMPTY_STR ([#228](https://github.com/pni-libraries/python-pninexus/pull/228])
- cast dims.size explicitly for osx ([#227](https://github.com/pni-libraries/python-pninexus/pull/227))
- add -fno-lto flag ([#225](https://github.com/pni-libraries/python-pninexus/pull/225))
- fix warning on stopping CI docker ([#221](https://github.com/pni-libraries/python-pninexus/pull/221))
- add debian 13 test ([#220](https://github.com/pni-libraries/python-pninexus/pull/220))
- technical release for libpninexus v3.4.0 (https://github.com/pni-libraries/libpninexus/issues/252) and libh5cpp v0.8.0 (https://github.com/ess-dmsc/h5cpp/issues/679)

## 3.3.0 - 2025-05-08
- fixes for writing utf8 strings ([#215](https://github.com/pni-libraries/python-pninexus/pull/215)), ([#213](https://github.com/pni-libraries/python-pninexus/pull/213)), ([#210](https://github.com/pni-libraries/python-pninexus/pull/210))
- technical release for libpninexus v3.3.0 (https://github.com/pni-libraries/libpninexus/issues/272) and libh5cpp v0.7.1 (https://github.com/ess-dmsc/h5cpp/issues/698)

## 3.2.3 - 2025-01-12
- adapt code to python 3.12 ([#201](https://github.com/pni-libraries/python-pninexus/pull/201))
- fix support for utf8 string as field/attrtirbute values ([#210](https://github.com/pni-libraries/python-pninexus/pull/210))

## 3.2.2 - 2023-10-12
- add pninexus filters to python wheel ([#196](https://github.com/pni-libraries/python-pninexus/pull/196))

## 3.2.1 - 2023-10-12
- replace README.md by README.rst ([#190](https://github.com/pni-libraries/python-pninexus/pull/190))
](https://github.com/pni-libraries/python-pninexus/pull/
## 3.2.0 - 2023-10-12
- string parameters to has/get_dataset/group/node added ([#185](https://github.com/pni-libraries/python-pninexus/pull/185))
- get_numpy_include_dirs to numpy.get_include changed ([#184](https://github.com/pni-libraries/python-pninexus/pull/183))
- technical release for libpninexus v3.2.0 ([#234](https://github.com/pni-libraries/libpninexus/issues/234)) and libh5cpp v0.6.0 (https://github.com/ess-dmsc/h5cpp/issues/631)

## 3.1.0 - 2023-04-25
- technical release for libpninexus v3.1.0 ([#196](https://github.com/pni-libraries/libpninexus/issues/213)) and libh5cpp v0.6.0 (https://github.com/ess-dmsc/h5cpp/issues/631)

## 3.0.3 - 2023-01-12
- technical release for libpninexus v3.0.3 ([#196](https://github.com/pni-libraries/libpninexus/issues/196)) and libh5cpp v0.5.2 (https://github.com/ess-dmsc/h5cpp/issues/616)

## 3.0.2 - 2023-01-05
- technical release for libpninexus v3.0.2 ([#196](https://github.com/pni-libraries/libpninexus/issues/196)) and libh5cpp v0.5.2 (https://github.com/ess-dmsc/h5cpp/issues/616)

## 3.0.1 - 2022-05-25
- technical release for libpninexus v3.0.1 ([#189](https://github.com/pni-libraries/libpninexus/issues/189)) and libh5cpp v0.5.1 (https://github.com/ess-dmsc/h5cpp/issues/602)

## 3.0.0 - 2022-05-09
- add documetation versioning ([#148](https://github.com/pni-libraries/python-pninexus/pull/148))
- switch tests to pytest ([#149](https://github.com/pni-libraries/python-pninexus/pull/149))
- add size, type and dimensions to Hyperslab ([#154](https://github.com/pni-libraries/python-pninexus/pull/154))
- update enums for SZip and ScaleOffset updated ([#157](https://github.com/pni-libraries/python-pninexus/pull/157))
- add PointsWrapper ([#160](https://github.com/pni-libraries/python-pninexus/pull/160))
- add H5CPP_ prefix ([#162](https://github.com/pni-libraries/python-pninexus/pull/162))
- use libpninexus 3.0.0 and libph5cpp 0.5.0 c++ libraries ([#166](https://github.com/pni-libraries/python-pninexus/pull/166))


## 2.0.0 - 2021-07-28
- use libpninexus 2.0.0 c++ libraries ([#143](https://github.com/pni-libraries/python-pninexus/pull/143))

## 1.3.4 - 2021-02-20
- HDF5_version to HDF5_Version attribute changed ([#132](https://github.com/pni-libraries/python-pninexus/pull/132))
- NeXus_version attribute removed ([#132](https://github.com/pni-libraries/python-pninexus/pull/132))
- h5cpp.current_library_version() function added ([#132](https://github.com/pni-libraries/python-pninexus/pull/132))

## 1.3.3 - 2021-01-27
- Dataset.fill_value methods added ([#127](https://github.com/pni-libraries/python-pninexus/pull/127))

## 1.3.2 - 2021-01-12
- ImageFlags wrapper added ([#125](https://github.com/pni-libraries/python-pninexus/pull/125))

## 1.3.1 - 2021-01-11
- python-pni changed to python-pninexus ([#121](https://github.com/pni-libraries/python-pninexus/pull/121))
- Path.absolute property fixed ([#120](https://github.com/pni-libraries/python-pninexus/pull/120))
- Path_is_root() method added ([#120](https://github.com/pni-libraries/python-pninexus/pull/120))

## 1.3.0 - 2021-01-08
- support for an image file buffer ([#99](https://github.com/pni-libraries/python-pninexus/pull/99))
- Integer class properties ([#100](https://github.com/pni-libraries/python-pninexus/pull/100))
- Float class properties ([#101](https://github.com/pni-libraries/python-pninexus/pull/101))
- Filter.is_encoding_enabled() and Filter.is_decoding_enabled methods ([#102](https://github.com/pni-libraries/python-pninexus/pull/102))
- ExternalFilters class ([#103](https://github.com/pni-libraries/python-pninexus/pull/103))
- NBit, SZip, ScaleOffset filters ([#104](https://github.com/pni-libraries/python-pninexus/pull/104))
- float16 type ([#105](https://github.com/pni-libraries/python-pninexus/pull/105))
- complex types ([#107](https://github.com/pni-libraries/python-pninexus/pull/107))
- Compound datatype class ([#107](https://github.com/pni-libraries/python-pninexus/pull/107))


