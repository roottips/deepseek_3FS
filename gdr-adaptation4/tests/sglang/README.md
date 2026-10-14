# Test artifacts for the SGLang GDR module.
#
# These are lightweight pytest unit tests. They do NOT require a real IB
# device or GPU; the verbs/CUDA surface is stubbed so the tests focus on
# the cache logic, index coalescing, and PeerInfo serialization that
# would otherwise be impossible to exercise in CI.
