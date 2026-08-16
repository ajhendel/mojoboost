"""The seven reserved objective codes, and the two things a reserved code
must get right before any trainer is connected to it.

This file was written on 2026-08-16 and for its first day could not run at
all: it imported `from testing` rather than `from std.testing` and had no
`def main()`, so it did not parse and the suite counted it as a test while
never evaluating a line of it. `tools/audit_test_structure.py` now gates
exactly that shape, and the freeze in `tools/api_snapshot.py` records this
file by name as the pin on the codes, so a deletion is a snapshot diff
rather than a silent loss of the only two-sided check on the numbers.

Run with `bash tools/run_tests.sh cpu test_objective_reserved_codes`.

What it pins:

1. **The numbers.** `QUERY_RMSE=13, PAIR_LOGIT=14, YETI_RANK=15, COX=16,
   SURVIVAL_AFT=17, MULTI_RMSE=-2, MULTI_RMSE_WITH_MISSING=-3`, and that they
   agree with the `comptime` constants still staged in the three modules that
   own the trainers. Those staged copies are the drift risk this creates:
   until `catboost_ranking.mojo`, `survival.mojo` and `multi_target.mojo`
   import their codes from here, there are two definitions of each number and
   nothing but this test makes them agree. An objective code is a number in a
   serialized model -- `serialize.save_model` writes it, `load_model` reads
   it -- so a disagreement is a model that loads as the wrong loss.

2. **That they are reserved and not reachable.** `objective_is_builtin` and
   `objective_is_known` must both be False for all seven, because every
   caller of those two is deciding whether a fit may proceed. Registering a
   code must not route a fit anywhere.

The registry facts each code carries -- task, link, init rule -- are checked
too, because those are exactly the facts a serialized model depends on and
they outlive the fit that wrote them. `SURVIVAL_AFT`'s `LINK_EXP` is the one
that mattered enough to register early: unregistered it fell through to
`LINK_IDENTITY`, which is right for `COX` and silently wrong for it.
"""

from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.objective_registry import (
    COX,
    INIT_CALLER,
    INIT_LINK_MEAN,
    INIT_ZERO,
    LINK_EXP,
    LINK_IDENTITY,
    MULTICLASS,
    MULTI_RMSE,
    MULTI_RMSE_WITH_MISSING,
    NAME_UNIMPLEMENTED,
    PAIR_LOGIT,
    QUERY_RMSE,
    SURVIVAL_AFT,
    TASK_RANKING,
    TASK_REGRESSION,
    YETI_RANK,
    objective_canonical_name,
    objective_init_kind,
    objective_is_builtin,
    objective_is_known,
    objective_is_multi_output,
    objective_link,
    objective_name_status,
    objective_needs_groups,
    objective_reserved,
    objective_reserved_trainer,
    objective_task,
)

# The staged copies, imported from the modules that own the trainers. This
# import is the point of the first test: if these three modules and the
# registry ever disagree about a number, this file stops compiling or stops
# passing, and nothing else in the tree would notice.
from mojotrees.catboost_ranking import (
    PAIR_LOGIT as STAGED_PAIR_LOGIT,
    QUERY_RMSE as STAGED_QUERY_RMSE,
    YETI_RANK as STAGED_YETI_RANK,
)
from mojotrees.survival import (
    COX as STAGED_COX,
    SURVIVAL_AFT as STAGED_SURVIVAL_AFT,
)
from mojotrees.multi_target import (
    MULTI_RMSE as STAGED_MULTI_RMSE,
    MULTI_RMSE_WITH_MISSING as STAGED_MULTI_RMSE_WITH_MISSING,
)


def test_the_numbers_are_the_numbers() raises:
    """The literal integers, spelled out. A renumber is a file-format break
    that raises nothing at load time, so it is pinned by value here and
    frozen again in `compatibility/api_snapshot.json`."""
    assert_equal(QUERY_RMSE, 13)
    assert_equal(PAIR_LOGIT, 14)
    assert_equal(YETI_RANK, 15)
    assert_equal(COX, 16)
    assert_equal(SURVIVAL_AFT, 17)
    assert_equal(MULTI_RMSE, -2)
    assert_equal(MULTI_RMSE_WITH_MISSING, -3)


def test_the_registry_and_the_staged_copies_agree() raises:
    """Two definitions of each number, one test that they are one number.

    16 and 17 rather than 13 and 14 is the scar: two lanes in one round
    assigned 13 and 14 to different objectives because each staged its codes
    in its own module, and what caught it was a lane's own unit test, not any
    gate. This is that test, made permanent and made two-sided.
    """
    assert_equal(QUERY_RMSE, STAGED_QUERY_RMSE)
    assert_equal(PAIR_LOGIT, STAGED_PAIR_LOGIT)
    assert_equal(YETI_RANK, STAGED_YETI_RANK)
    assert_equal(COX, STAGED_COX)
    assert_equal(SURVIVAL_AFT, STAGED_SURVIVAL_AFT)
    assert_equal(MULTI_RMSE, STAGED_MULTI_RMSE)
    assert_equal(
        MULTI_RMSE_WITH_MISSING, STAGED_MULTI_RMSE_WITH_MISSING
    )


def test_no_two_objective_codes_collide() raises:
    """Every code in the space, compared pairwise. The check that would have
    caught the 13/14 collision without anyone having to think of it."""
    var codes: List[Int] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        QUERY_RMSE, PAIR_LOGIT, YETI_RANK, COX, SURVIVAL_AFT,
        MULTICLASS, MULTI_RMSE, MULTI_RMSE_WITH_MISSING,
    ]
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            assert_true(
                codes[i] != codes[j],
                String(
                    "objective codes ", i, " and ", j,
                    " share the value ", codes[i],
                ),
            )


def test_reserved_is_not_known_and_not_builtin() raises:
    """Registering a code must not route a fit anywhere.

    Every caller of `objective_is_builtin` and `objective_is_known` is
    deciding whether a fit may proceed, and the answer for all seven is no
    until a binding entry point exists. This is the assertion that keeps the
    registry addition from becoming the accepted-and-silently-ignored defect
    it was written to prevent.
    """
    var reserved: List[Int] = [
        QUERY_RMSE, PAIR_LOGIT, YETI_RANK, COX, SURVIVAL_AFT,
        MULTI_RMSE, MULTI_RMSE_WITH_MISSING,
    ]
    for i in range(len(reserved)):
        var code = reserved[i]
        assert_true(objective_reserved(code), String("reserved ", code))
        assert_false(objective_is_builtin(code), String("builtin ", code))
        assert_false(objective_is_known(code), String("known ", code))
        assert_true(
            objective_reserved_trainer(code).byte_length() > 0,
            String("code ", code, " must name its trainer"),
        )
    # And the predicate is exact: a connected code is not reserved.
    assert_false(objective_reserved(0))
    assert_false(objective_reserved(MULTICLASS))
    assert_equal(objective_reserved_trainer(0), String(""))


def test_each_reserved_code_round_trips_under_a_name() raises:
    """A serialized model reports the loss it was trained with. These are the
    names it reports."""
    assert_equal(objective_canonical_name(QUERY_RMSE), String("query_rmse"))
    assert_equal(objective_canonical_name(PAIR_LOGIT), String("pair_logit"))
    assert_equal(objective_canonical_name(YETI_RANK), String("yeti_rank"))
    assert_equal(objective_canonical_name(COX), String("cox"))
    assert_equal(
        objective_canonical_name(SURVIVAL_AFT), String("survival_aft")
    )
    assert_equal(objective_canonical_name(MULTI_RMSE), String("multi_rmse"))
    assert_equal(
        objective_canonical_name(MULTI_RMSE_WITH_MISSING),
        String("multi_rmse_with_missing"),
    )


def test_survival_aft_is_the_link_that_had_to_be_registered_early() raises:
    """`SurvivalAft`'s raw score is the log of the survival time, so its
    inverse link is `exp`. Unregistered it fell through to `LINK_IDENTITY`,
    which is correct for `COX` and silently wrong for it -- a prediction off
    by an exponential with nothing raised. `survival.survival_aft_predicted_
    time` existed only as the workaround."""
    assert_equal(objective_link(SURVIVAL_AFT), LINK_EXP)
    assert_equal(objective_link(COX), LINK_IDENTITY)
    assert_equal(objective_link(QUERY_RMSE), LINK_IDENTITY)
    assert_equal(objective_link(PAIR_LOGIT), LINK_IDENTITY)
    assert_equal(objective_link(YETI_RANK), LINK_IDENTITY)
    assert_equal(objective_link(MULTI_RMSE), LINK_IDENTITY)


def test_task_and_grouping() raises:
    """The three CatBoost ranking losses are ranking tasks and need query
    groups; the survival and multi-output codes are regression and do not."""
    assert_equal(objective_task(QUERY_RMSE), TASK_RANKING)
    assert_equal(objective_task(PAIR_LOGIT), TASK_RANKING)
    assert_equal(objective_task(YETI_RANK), TASK_RANKING)
    assert_equal(objective_task(COX), TASK_REGRESSION)
    assert_equal(objective_task(SURVIVAL_AFT), TASK_REGRESSION)
    assert_equal(objective_task(MULTI_RMSE), TASK_REGRESSION)
    assert_true(objective_needs_groups(QUERY_RMSE))
    assert_true(objective_needs_groups(PAIR_LOGIT))
    assert_true(objective_needs_groups(YETI_RANK))
    assert_false(objective_needs_groups(COX))
    assert_false(objective_needs_groups(MULTI_RMSE))


def test_init_rules_come_from_the_trainers() raises:
    """Each taken from the trainer that would run, not guessed.

    All three ranking losses and `COX` boost from zero because their level is
    unidentifiable -- QueryRMSE subtracts the query mean, the pairwise losses
    see only differences, and adding a constant to every Cox raw score leaves
    the partial likelihood exactly unchanged. `train_survival_aft` takes a
    caller-supplied `base_score`, which is `INIT_CALLER`. `MULTI_RMSE` is
    squared error per plane.
    """
    assert_equal(objective_init_kind(QUERY_RMSE), INIT_ZERO)
    assert_equal(objective_init_kind(PAIR_LOGIT), INIT_ZERO)
    assert_equal(objective_init_kind(YETI_RANK), INIT_ZERO)
    assert_equal(objective_init_kind(COX), INIT_ZERO)
    assert_equal(objective_init_kind(SURVIVAL_AFT), INIT_CALLER)
    assert_equal(objective_init_kind(MULTI_RMSE), INIT_LINK_MEAN)


def test_multi_rmse_is_multi_output() raises:
    """`train_multi_rmse` grows one tree per output plane per round, which is
    why its codes are negative at all."""
    assert_true(objective_is_multi_output(MULTI_RMSE))
    assert_true(objective_is_multi_output(MULTI_RMSE_WITH_MISSING))
    assert_true(objective_is_multi_output(MULTICLASS))
    assert_false(objective_is_multi_output(COX))


def test_the_names_refuse_with_the_trainer_rather_than_resolve() raises:
    """Asking for one of these by name must NOT resolve to a code.

    This is the half that keeps the registry addition honest. If
    `objective_code_from_name("cox")` returned 16, an estimator would accept
    `objective="cox"` and hand a trainer a number it cannot fit. Instead the
    names land on the existing "real thing, not implemented" path, so the
    message names `survival.train_cox` and the missing call -- and
    `objective_name_status` reports `NAME_UNIMPLEMENTED` rather than
    `NAME_SUPPORTED`, which is the query a Python estimator makes.
    """
    from mojotrees.objective_registry import objective_code_from_name

    for name in [
        String("cox"),
        String("survival_aft"),
        String("queryrmse"),
        String("pairlogit"),
        String("yetirank"),
        String("multi_rmse"),
    ]:
        assert_equal(objective_name_status(name), NAME_UNIMPLEMENTED)
        with assert_raises(contains="is not implemented"):
            _ = objective_code_from_name(name)
    # And the reason names the module and the function, not "not implemented".
    with assert_raises(contains="survival.mojo"):
        _ = objective_code_from_name(String("cox"))
    with assert_raises(contains="train_catboost_ranker"):
        _ = objective_code_from_name(String("yetirank"))
    with assert_raises(contains="train_multi_rmse"):
        _ = objective_code_from_name(String("multi_rmse"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
