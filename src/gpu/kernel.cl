// b20crunch GPU kernel. Real entry points land with the keccak port; this
// minimal kernel exists so the embed/compile/error path is testable now.
__kernel void probe(__global uint *out) {
    if (get_global_id(0) == 0)
        out[0] = 0xb20;
}
