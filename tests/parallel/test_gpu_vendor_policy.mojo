"""The CUDA and HIP policies of gpu_vendor_policy.mojo through one set of
functions, driven by `VendorTraits`.

What this pins:

- The two traits carry the documented vendor numbers, and `traits_for`
  refuses an API that is neither.
- An unreported device plans exactly as the portable path does on both
  vendors (the plan matches its own baseline).
- The subgroup admissibility rule is the vendor's: CUDA admits 32 alone,
  HIP admits 64 and 32, and both refuse a width outside their set while
  passing an unreported one.
- The reported width refines the block granularity, and the packed-body
  transaction count is measured against the vendor's transaction size.
- No strategy preference is invented on either vendor.

Everything here is host arithmetic; no device is opened.
"""

from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees.apple_gpu_policy import API_CUDA, API_HIP, API_METAL
from mojotrees.gpu_histogram_specializations import (
    PACK_LANES,
    WINDOW_OK,
    KernelFeatures,
    PackedLoadWindow,
)
from mojotrees.gpu_tiling import STRATEGY_AUTO, WARP_GRANULARITY
from mojotrees.gpu_vendor_policy import (
    AMD_CACHE_LINE_BYTES,
    AMD_LDS_PER_WORKGROUP_BYTES,
    CUDA_SECTOR_BYTES,
    CUDA_STATIC_SHARED_CEILING_BYTES,
    DeviceReport,
    VendorTraits,
    block_width_granularity,
    coalescing_bytes,
    cuda_traits,
    derive_vendor_plan,
    describe_vendor,
    hip_traits,
    packed_body_transactions,
    require_subgroup_width_plausible,
    static_shared_ceiling,
    subgroup_matches_granularity,
    traits_for,
    vendor_preferred_strategy,
    vendor_specialization,
)


def _raises(fn_ok: Bool) -> Bool:
    return not fn_ok


def test_traits_carry_the_vendor_numbers() raises:
    var cuda = cuda_traits()
    var hip = hip_traits()
    assert_equal(cuda.api, API_CUDA)
    assert_equal(hip.api, API_HIP)
    assert_equal(static_shared_ceiling(cuda), CUDA_STATIC_SHARED_CEILING_BYTES)
    assert_equal(static_shared_ceiling(hip), AMD_LDS_PER_WORKGROUP_BYTES)
    assert_equal(coalescing_bytes(cuda), CUDA_SECTOR_BYTES)
    assert_equal(coalescing_bytes(hip), AMD_CACHE_LINE_BYTES)
    assert_equal(len(cuda.subgroup_widths), 1)
    assert_equal(len(hip.subgroup_widths), 2)
    assert_equal(traits_for(API_CUDA).api, API_CUDA)
    assert_equal(traits_for(API_HIP).api, API_HIP)
    var refused = False
    try:
        _ = traits_for(API_METAL)
    except:
        refused = True
    assert_true(refused)


def test_unreported_device_plans_as_the_portable_path() raises:
    var report = DeviceReport.unreported()
    var compiled = KernelFeatures.none()
    var apis = [API_CUDA, API_HIP]
    for i in range(len(apis)):
        var traits = traits_for(apis[i])
        var plan = derive_vendor_plan(
            traits, report, 100_000, 8, 64, compiled
        )
        assert_true(plan.matches_baseline())
        assert_equal(plan.api, apis[i])
        assert_equal(block_width_granularity(report), WARP_GRANULARITY)
        assert_true(subgroup_matches_granularity(report))
        assert_equal(vendor_preferred_strategy(plan), STRATEGY_AUTO)
        var spec = vendor_specialization(traits, report, compiled)
        assert_false(spec.subgroup_width_reported)
        assert_false(spec.concurrent_queues)
        assert_false(spec.unified_memory_inferable)
        var line = describe_vendor(traits, report, compiled)
        assert_true(line.byte_length() > 0)


def _plausible(traits: VendorTraits, warp: Int) -> Bool:
    var report = DeviceReport.queried(warp_size=warp)
    try:
        require_subgroup_width_plausible(traits, report)
        return True
    except:
        return False


def test_subgroup_admissibility_is_the_vendors() raises:
    var cuda = cuda_traits()
    var hip = hip_traits()
    assert_true(_plausible(cuda, 0))
    assert_true(_plausible(cuda, 32))
    assert_false(_plausible(cuda, 64))
    assert_true(_plausible(hip, 0))
    assert_true(_plausible(hip, 32))
    assert_true(_plausible(hip, 64))
    assert_false(_plausible(hip, 48))


def test_reported_width_refines_granularity() raises:
    var rdna = DeviceReport.queried(warp_size=32)
    var cdna = DeviceReport.queried(warp_size=64)
    assert_equal(block_width_granularity(rdna), 32)
    assert_equal(block_width_granularity(cdna), 64)
    assert_false(subgroup_matches_granularity(rdna))
    assert_true(subgroup_matches_granularity(cdna))


def test_packed_body_is_measured_against_the_vendor_transaction() raises:
    var window = PackedLoadWindow(True, 0, 16, 0, WINDOW_OK)
    var body_bytes = 16 * PACK_LANES
    assert_equal(
        packed_body_transactions(cuda_traits(), window),
        (body_bytes + CUDA_SECTOR_BYTES - 1) // CUDA_SECTOR_BYTES,
    )
    assert_equal(
        packed_body_transactions(hip_traits(), window),
        (body_bytes + AMD_CACHE_LINE_BYTES - 1) // AMD_CACHE_LINE_BYTES,
    )
    var unusable = PackedLoadWindow(False, 3, 0, 0, WINDOW_OK)
    assert_equal(packed_body_transactions(hip_traits(), unusable), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
