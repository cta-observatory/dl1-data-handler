try:
    from ctao_datamodel import __version__ as ctao_datamodel_version
except ImportError:
    CTAO_DATAMODEL_VERSION = None
    CTAO_DATAMODEL_GE_1_4_0 = False
else:
    CTAO_DATAMODEL_VERSION = tuple(
        int(v) for v in ctao_datamodel_version.split(".")[:3]
    )
    CTAO_DATAMODEL_GE_1_4_0 = CTAO_DATAMODEL_VERSION >= (1, 4, 0)
