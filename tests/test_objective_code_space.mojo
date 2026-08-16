"""The objective code space as a whole: no integer names two losses, and
every code round-trips through the name it is saved under.

This is the gate the 13/14 collision got past. Two lanes in one round
independently assigned 13 and 14 to different objectives, each staging its
codes in its own module, and what caught it was one lane's own unit test
pinning its own values, not any check over the space. A per-lane test cannot
catch a collision between lanes, by construction: it only knows its own
numbers.

So this file walks lists rather than literals. `all_assigned_objective_codes`
is the whole space, connected codes and reserved codes together, and a code
added anywhere lands in it or lands nowhere; the duplicate check and the
round-trip check both read it. tests/test_objective_reserved_codes.mojo pins
the literal integers and the staged copies in the three trainer modules, and
this file deliberately does not repeat either: pinning a value twice is two
places to edit, and the thing that was missing was never the pin.

The round trip is name -> code -> canonical name -> code, closed. It is the
weaker-looking half and it is the half that was actually broken:
`MULTI_RMSE_WITH_MISSING` had a canonical name that `objective_canonical_
name` reported and that no name function accepted, so `objective_name_status`
called it unknown and the Python estimator's message for it was the
unknown-name one. A code whose own name does not resolve is a model file
whose loss field cannot be read back by name.

RUN, once, on its own, under the round's one-test budget:
`MOJOTREES_TEST_PKG=0 tools/run_tests.sh cpu test_objective_code_space` ->
`ok test_objective_code_space (2s wall, 0.113ms in tests)`. The `PKG=0` is
not a preference: `mojo precompile -I src src/mojotrees` fails at head on
`src/mojotrees/trainset.mojo:1275`, which is nothing to do with this file
and blocks the default mode for the whole suite.
"""

from std.testing import assert_equal, assert_raises, assert_true
from std.testing import TestSuite

from mojotrees.objective_registry import (
    COX,
    MULTI_RMSE,
    MULTI_RMSE_WITH_MISSING,
    NAME_SUPPORTED,
    NAME_UNIMPLEMENTED,
    NO_OBJECTIVE_CODE,
    PAIR_LOGIT,
    QUERY_RMSE,
    SURVIVAL_AFT,
    YETI_RANK,
    all_assigned_objective_codes,
    all_objective_codes,
    all_reserved_objective_codes,
    any_objective_code_from_name,
    check_objective_codes_distinct,
    objective_alias_names,
    objective_canonical_name,
    objective_code_from_name,
    objective_is_known,
    objective_name_status,
    objective_reserved,
    objective_unimplemented_canonical,
    objective_unimplemented_reason,
    reserved_objective_alias_names,
    reserved_objective_code_from_name,
    unimplemented_objective_alias_names,
)


def test_no_integer_names_two_objectives() raises:
    """The check that the 13/14 collision needed and did not have.

    Pairwise over the whole assigned space, read from the registry's own
    enumeration rather than typed out here, so the next code assigned is
    covered by this test the moment it is added to the list and by nothing
    at all if it is not.
    """
    check_objective_codes_distinct()
    # And the same statement without going through the registry's function,
    # so a `check_objective_codes_distinct` that silently returned early
    # would not take this assertion with it.
    var codes = all_assigned_objective_codes()
    assert_equal(len(codes), 21)
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            assert_true(
                codes[i] != codes[j],
                String(
                    "objective code ",
                    codes[i],
                    " appears at positions ",
                    i,
                    " and ",
                    j,
                ),
            )


def test_the_assigned_space_is_the_two_halves_and_nothing_else() raises:
    """`all_assigned_objective_codes` is exactly the connected codes plus
    the reserved ones, and the split agrees with `objective_reserved`.

    A code that drifted into one list and not the other would make the
    duplicate check above miss it, which is the failure mode this file
    exists to prevent, so the composition is asserted rather than assumed.
    """
    var connected = all_objective_codes()
    var reserved = all_reserved_objective_codes()
    var assigned = all_assigned_objective_codes()
    assert_equal(len(assigned), len(connected) + len(reserved))
    for i in range(len(connected)):
        assert_equal(assigned[i], connected[i])
        assert_true(
            objective_is_known(connected[i]),
            String("connected code ", connected[i], " must be known"),
        )
        assert_true(
            not objective_reserved(connected[i]),
            String("connected code ", connected[i], " must not be reserved"),
        )
    for i in range(len(reserved)):
        assert_equal(assigned[len(connected) + i], reserved[i])
        assert_true(
            objective_reserved(reserved[i]),
            String("code ", reserved[i], " must be reserved"),
        )
        assert_true(
            not objective_is_known(reserved[i]),
            String("reserved code ", reserved[i], " must not be known"),
        )


def test_every_code_round_trips_through_its_name() raises:
    """Code -> canonical name -> code, for every assigned code.

    The direction that was broken. `objective_canonical_name` answered for
    all twenty-one and only fourteen of the names resolved back, so seven
    codes had a name that named nothing.
    """
    var codes = all_assigned_objective_codes()
    for i in range(len(codes)):
        var name = objective_canonical_name(codes[i])
        assert_true(
            name.byte_length() > 0,
            String("code ", codes[i], " has no canonical name"),
        )
        assert_equal(
            any_objective_code_from_name(name),
            codes[i],
            String("code ", codes[i], " does not round trip via '", name, "'"),
        )


def test_every_name_round_trips_through_its_code() raises:
    """Name -> code -> canonical name -> code, for every spelling either
    resolver accepts. The other direction, and the one that catches an alias
    pointing at the wrong code rather than a code with no alias."""
    var names = objective_alias_names()
    for i in range(len(names)):
        var code = objective_code_from_name(names[i])
        var canonical = objective_canonical_name(code)
        assert_equal(
            any_objective_code_from_name(canonical),
            code,
            String("alias '", names[i], "' does not round trip"),
        )
        assert_equal(objective_name_status(names[i]), NAME_SUPPORTED)

    var reserved_names = reserved_objective_alias_names()
    for i in range(len(reserved_names)):
        var code = reserved_objective_code_from_name(reserved_names[i])
        var canonical = objective_canonical_name(code)
        assert_equal(
            any_objective_code_from_name(canonical),
            code,
            String("reserved '", reserved_names[i], "' does not round trip"),
        )
        assert_true(
            objective_reserved(code),
            String("'", reserved_names[i], "' resolved to a live code"),
        )


def test_the_reserved_names_are_one_set_not_two() raises:
    """Every reserved spelling both resolves to a code and reports a reason.

    Two chains list these names -- `_reserved_objective_code` and
    `objective_unimplemented_canonical` -- and a name in one and not the
    other is either a code with no message or a message with no code. The
    first is what shipped: `multi_rmse_with_missing` had a code and no
    message, so the Python estimator called it an unknown name.
    """
    var reserved_names = reserved_objective_alias_names()
    var unimplemented = unimplemented_objective_alias_names()
    for i in range(len(reserved_names)):
        var name = reserved_names[i]
        assert_true(
            objective_unimplemented_canonical(name).byte_length() > 0,
            String("reserved '", name, "' reports no canonical name"),
        )
        assert_true(
            objective_unimplemented_reason(name).byte_length() > 0,
            String("reserved '", name, "' reports no reason"),
        )
        assert_equal(objective_name_status(name), NAME_UNIMPLEMENTED)
        var found = False
        for j in range(len(unimplemented)):
            if unimplemented[j] == name:
                found = True
        assert_true(
            found,
            String(
                "reserved '",
                name,
                "' is missing from UNIMPLEMENTED_OBJECTIVE_ALIAS_NAMES, so"
                " python/mojotrees/_fit_args.py cannot quote its reason",
            ),
        )
        # And the reason names the module, which is the whole point of a
        # reserved code reporting differently from an unimplemented one.
        assert_true(
            objective_unimplemented_reason(name).find(".mojo") >= 0,
            String("reserved '", name, "' does not name its module"),
        )


def test_catboost_capitalization_resolves_once_lowercased() raises:
    """CatBoost's own spellings, lowercased by the caller as everywhere in
    this module (`python/mojotrees/_fit_args.py` does `name.strip().lower()`
    before it asks). A ported CatBoost script writes `QueryRMSE`, a mojotrees
    model file writes `query_rmse`, and both must land on 13."""
    assert_equal(any_objective_code_from_name(String("queryrmse")), QUERY_RMSE)
    assert_equal(any_objective_code_from_name(String("pairlogit")), PAIR_LOGIT)
    assert_equal(any_objective_code_from_name(String("yetirank")), YETI_RANK)
    assert_equal(any_objective_code_from_name(String("cox")), COX)
    assert_equal(
        any_objective_code_from_name(String("survivalaft")), SURVIVAL_AFT
    )
    assert_equal(any_objective_code_from_name(String("multirmse")), MULTI_RMSE)
    assert_equal(
        any_objective_code_from_name(String("multirmsewithmissingvalues")),
        MULTI_RMSE_WITH_MISSING,
    )


def test_the_fit_gate_still_refuses_every_reserved_name() raises:
    """The identity resolver must not have widened the fit gate.

    `objective_code_from_name` is what an estimator calls before it hands a
    trainer a number. If it resolved `cox` to 16, a fit would be routed to a
    trainer nothing connects. It does not, and adding
    `any_objective_code_from_name` beside it must not change that, which is
    what makes the two-function split worth having rather than a rename.
    """
    var reserved_names = reserved_objective_alias_names()
    for i in range(len(reserved_names)):
        with assert_raises(contains="is not implemented"):
            _ = objective_code_from_name(reserved_names[i])


def test_an_unimplemented_lightgbm_name_still_has_no_code() raises:
    """`multiclassova` and friends are real objectives with no code at all,
    which is a different thing from a code no trainer reaches. The identity
    resolver must refuse them too, or the round trip above would be closing
    over names that name nothing."""
    var names: List[String] = [
        String("multiclassova"),
        String("rank_xendcg"),
        String("cross_entropy_lambda"),
        String("xentlambda"),
    ]
    for i in range(len(names)):
        with assert_raises(contains="is not implemented"):
            _ = any_objective_code_from_name(names[i])
        with assert_raises(contains="not a reserved objective name"):
            _ = reserved_objective_code_from_name(names[i])


def test_the_no_code_sentinel_is_outside_the_assigned_space() raises:
    """`NO_OBJECTIVE_CODE` is what the reserved chain returns for a name it
    does not know, so it must never equal a code. It is frozen in
    compatibility/api_snapshot.json beside the codes for the same reason."""
    var codes = all_assigned_objective_codes()
    for i in range(len(codes)):
        assert_true(
            codes[i] != NO_OBJECTIVE_CODE,
            String("code ", codes[i], " collides with NO_OBJECTIVE_CODE"),
        )
    with assert_raises(contains="not a reserved objective name"):
        _ = reserved_objective_code_from_name(String("no_such_objective"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
