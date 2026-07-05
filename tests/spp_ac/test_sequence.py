from spp_ac.env.sequence import StowageSequence


def test_sequence_length():
    seq = StowageSequence(R_left=3, R_right=3, T=2)
    total_slots = 3 * 2 + 3 * 2
    assert len(seq) == 12


def test_sequence_skip_non_loadable():
    seq = StowageSequence(R_left=2, R_right=2, T=1,
                           non_loadable={(0, 1, 0), (1, 1, 0)})
    assert len(seq) == 2


def test_sequence_bottom_up_both_sides():
    seq = StowageSequence(R_left=2, R_right=2, T=2)
    slots = [seq[i] for i in range(len(seq))]
    assert slots[0] == (0, 0, 0)
    assert slots[1] == (1, 1, 0)


def test_paired_slot_40ft():
    seq = StowageSequence(R_left=2, R_right=2, T=1)
    paired = seq.get_paired_slot(0)
    assert paired is not None
    assert paired[0] == 1
    assert paired[1] == 0
    assert paired[2] == 0


def test_mark_occupied_skips_paired():
    seq = StowageSequence(R_left=2, R_right=2, T=2)
    seq.mark_occupied(0)
    assert len(seq) == 6
