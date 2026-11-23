include "hash/blake3/linearhash.circom";

// AggregatePublics for the PCI join: an ordered, count-tagged accumulator over
// the child digests. (na, A) + (nb, B) -> (na + nb, LinearHash(na, A[4], nb, B[4])).
// The client recomputes it over the receipts in roster order with the canonical
// balanced shape (pci_client::accumulator::fold) and runs the whole join natively.
template AggregatePublics(nPublicsAgg) {
    signal output aggregated_publics[nPublicsAgg];
    signal input a_publics[ZISK_PUBLICS()];
    signal input b_publics[ZISK_PUBLICS()];

    assert(nPublicsAgg == 5);
    for (var i = 5; i < ZISK_PUBLICS(); i++) {
        a_publics[i] === 0;
        b_publics[i] === 0;
    }
    component h = LinearHash(10, 2, 1);
    for (var k = 0; k < 5; k++) {
        h.in[k][0] <== a_publics[k];
        h.in[5 + k][0] <== b_publics[k];
    }
    aggregated_publics[0] <== a_publics[0] + b_publics[0];
    for (var k = 0; k < 4; k++) {
        aggregated_publics[1 + k] <== h.out[k];
    }
}
