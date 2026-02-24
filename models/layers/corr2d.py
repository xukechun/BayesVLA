import traceback


try:
    from ._corr_scs import CorrKernel

except Exception as e:
    print("[ERR ] Error when importing `spatial_correlation_sampler`")
    traceback.print_exc()
    print("[INFO] Switch to the taichi version")

    from ._corr_ti import CorrKernel


__all__ = ["CorrKernel"]

