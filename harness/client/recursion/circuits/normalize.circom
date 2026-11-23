include "hash/blake3/linearhash.circom";

// NormalizePublics for the PCI join. A leaf's 64 publics are the 32-byte
// receipt digest as 8 little-endian u32 limbs, then zeros (child ABI v1).
// Output: (count = 1, acc = LinearHash(8 limbs)), zeros after slot 5. Must be
// satisfiable on aggregates too (their slots 5.. are zero, so slots 8.. are).
// Native counterpart: pci_client::accumulator::leaf.
template NormalizePublics() {
    signal input publics[ZISK_PUBLICS()];
    signal input programIndex;
    signal output recurser_publics[ZISK_PUBLICS()];

    for (var i = 8; i < ZISK_PUBLICS(); i++) {
        publics[i] === 0;
    }
    component h = LinearHash(8, 2, 1);
    for (var i = 0; i < 8; i++) {
        h.in[i][0] <== publics[i];
    }
    recurser_publics[0] <== 1;
    for (var k = 0; k < 4; k++) {
        recurser_publics[1 + k] <== h.out[k];
    }
    for (var i = 5; i < ZISK_PUBLICS(); i++) {
        recurser_publics[i] <== 0;
    }
    // Role binding is by receipt-digest domain (each approved program only
    // emits its own role's encoding); programIndex is not needed for it.
    _ <== programIndex;
}
