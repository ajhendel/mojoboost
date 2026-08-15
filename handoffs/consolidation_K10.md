# consolidation K10 - orphan feature dispositions and duplicate rewires

Run 2026-08-15 by lane K10 of the consolidation round (coordinator record:
handoffs/consolidation_round.md). No feature work, nothing wired into an
entry point, behavior frozen.

## Disposition table (docs/INTEGRATION_INVENTORY.md style)

| module | lines | verdict | detail |
|---|---|---|---|
| cegb | 1174 | parked | LightGBM cegb_* controls, complete and self-contained; nothing imports it. Unblocked by a trainer accepting the cegb params (tree_parameters_extra.check_extra_option_supported still refuses cegb_penalty_feature_lazy) and a boosting-loop hook. |
| linear_tree | 2322 | parked | linear_tree=true (LinearEnsemble). Unblocked by Booster holding a LinearEnsemble and model I/O carrying it. Codes and renewal rule now come from objective_registry (e3ed5ab). |
| model_editing | 2235 | parked | In-place leaf editing. Its MODEL_EDITING_SUPPORTED = True is the feature's claim; inspection.mojo's False is what ships. Unblocked by connecting it, at which point inspection's flag and status JSON must be replaced by this module's (see deferred). |
| ranking_advanced | 2150 | parked | Position bias, pair sampling, custom label gain, fold shuffle. Unblocked by fit_ranker growing those params. RNG now from rng.mojo (747b2fc). |
| validation | 1924 | owned-by-lane remaining_12 | Central validation layer; python/_validation.py manifest names its functions by string ("validation.check_relevance_labels"). Unblocked by the manifest lane routing trainers through it. |
| lgbm_model_io | 2869 | parked | LightGBM text model read/write, EXPERIMENTAL and quarantined by its own LGBM_INTEROP_STATUS; reached only from tests/parallel/test_lgbm_model_io.mojo. Unblocked by a binding plus the interop status flipping. |
| gpu_categorical | 852 | parked | GPU category statistics; the GPU trainer refuses categoricals. Unblocked by train_gpu accepting categorical specs. |
| gpu_sparse | 1736 | parked | Reached only from gpu_categorical. Same unblocker. |
| gpu_sparse_layout | 1081 | parked | Reached only from gpu_sparse. Same unblocker. |
| external_memory | 2199 | parked | Streaming Dataset construction over sequence.mojo (K4 parked that too). Unblocked by an export plus a chunk binding; connect together with sequence. |
| alternate_boosting | 1110 | owned-by-lane connect_17 | DART and RF dispatch; boosting.mojo still owns the only round loop. Unblocked by a boosting= parameter routing to it. |
| boosting_dart | 714 | owned-by-lane connect_17 | Reached only from alternate_boosting. RNG now from rng.mojo (c6a39ae). |
| boosting_rf | 1613 | owned-by-lane connect_17 | Reached only from alternate_boosting. LAMBDARANK now from objective_registry (26f04f3, 8b8cc53). |
| initialization | 911 | connected | src/mojotrees/__init__.mojo, device_policy.mojo, gpu_runtime.mojo, bindings/basic_bindings.mojo. |
| raw_data | 264 | connected | src/mojotrees/__init__.mojo, trainset.mojo, bindings/dataset_bindings.mojo (also external_memory, sequence). |
| model_dump | 854 | connected | inspection.mojo, bindings/inspection_bindings.mojo. _MAX_CATEGORY now from categorical (5f21ac8). |

Nothing met the deletion bar; no module was deleted.

## Duplicate rewires (orphan imports authority, bit-exact)

| pair | resolution | commit | check |
|---|---|---|---|
| splitmix64/uniform/GOLDEN in boosting_dart | import from rng.mojo; private mixer, _uniform, _GOLDEN, _TWO_POW_NEG_53 deleted; _stream derivation kept | c6a39ae | scratchpad check_k10.mojo: dart _stream(seed, round) == splitmix64(masked_seed ^ round*GOLDEN) computed directly |
| _splitmix64/_uniform in model_editing (imported from sampling, which K1 removed: module did not compile) | import from rng.mojo | 96184c5 | same check file, compiles and _is_finite behaves |
| _GOLDEN + sampling imports in ranking_advanced (same broken import) | import GOLDEN, splitmix64, uniform from rng.mojo | 747b2fc | _pair_kept(stream,i,j,rate) == uniform(splitmix64(stream ^ i*GOLDEN)+j) < rate |
| _F64_MAX inspection vs model_editing | model_editing imports inspection._F64_MAX | 96184c5 | compile |
| _QUANTILE/_L1/_LAMBDARANK/_MAPE + _objective_renews_leaves mirror in linear_tree | import from objective_registry (imports only metrics.mojo, so linear_tree stays below boosting.mojo as its header requires) | e3ed5ab | mirrors == registry values; renews rule agrees on QUANTILE, L1, MAPE |
| LeafStats gpu_frontier vs linear_tree | NOT identical (3 scalars vs centered second-moment matrix record); linear_tree's renamed LinearLeafStats, 5 internal sites | e3ed5ab | compile |
| _LAMBDARANK mirror in boosting_rf | import LAMBDARANK from objective_registry, unaliased | 26f04f3, 8b8cc53 | value == 7 |
| _MAX_CATEGORY categorical vs model_dump | model_dump imports categorical._MAX_CATEGORY | 5f21ac8 | value == 1<<31 |

Test budget used: one scratchpad compile-and-value check (imports all seven
touched modules) run twice; no repository test reaches any of these modules,
so none exists to move. Duplicate section: 32 -> 27. Orphan section: 19 both
before and after (parking is the verdict, as expected).

## Deferred, with reasons

- MODEL_EDITING_SUPPORTED and model_editing_status_json (inspection vs
  model_editing): the two definitions DISAGREE on purpose (False ships,
  True is the parked feature). Importing either way changes a stated fact.
  Resolve when model_editing is connected: inspection's flag and status
  become re-exports of model_editing's. Not touched.
- check_relevance_labels (ranking_advanced vs validation): a name collision,
  not a duplicate; different signatures (Int labels vs LabelGain table, vs
  Float64 column to Int list). Renaming validation's would break the string
  reference in python/mojotrees/_validation.py:592, which K10 may not edit.
  Left for the validation manifest lane; suggested rename on that side:
  validation.relevance_labels_from_column.
- CancelToken (sequence vs validation): K4 recorded them as not identical;
  not merged. Owner is whichever of sequence / validation connects first.
- objective_renews_leaves boosting vs objective_registry: documented wrapper;
  coordinator's call (see consolidation_round.md).
- The `_stream(seed, index)` derivation is still byte-identical in bagging,
  goss, and boosting_dart (K1 noted this); a shared rng.index_stream would
  need rng.mojo, which K10 may not edit.

## CLASSIFICATION block for tools/connectivity_audit.py (coordinator applies)

```python
    "cegb": (
        PENDING,
        "consolidation_K10",
        "LightGBM cegb_* controls, complete and self-contained. Parked until "
        "a trainer accepts the cegb params and the boosting loop hooks it.",
    ),
    "linear_tree": (
        PENDING,
        "consolidation_K10",
        "linear_tree=true ensembles. Parked until Booster holds a "
        "LinearEnsemble and model I/O carries it; codes come from "
        "objective_registry.",
    ),
    "model_editing": (
        PENDING,
        "consolidation_K10",
        "In-place leaf editing. Its MODEL_EDITING_SUPPORTED=True is the "
        "feature's claim; inspection.mojo's False is what ships. Parked "
        "until connected, when inspection re-exports this module's status.",
    ),
    "ranking_advanced": (
        PENDING,
        "consolidation_K10",
        "Position bias, pair sampling, custom label gain, fold shuffle. "
        "Parked until fit_ranker grows those parameters.",
    ),
    "validation": (
        PENDING,
        "remaining_12",
        "Central validation layer named by python _validation.py's manifest. "
        "Parked until the manifest lane routes trainers through it.",
    ),
    "lgbm_model_io": (
        EXPERIMENTAL,
        "consolidation_K10",
        "LightGBM text model interop, quarantined by its own "
        "LGBM_INTEROP_STATUS and reached only from its test. Parked until a "
        "binding exists and the status flips.",
    ),
    "gpu_categorical": (
        PENDING,
        "consolidation_K10",
        "GPU category statistics; the GPU trainer refuses categoricals. "
        "Parked until train_gpu accepts categorical specs.",
    ),
    "gpu_sparse": (
        PENDING,
        "consolidation_K10",
        "Reached only from gpu_categorical; same unblocker.",
    ),
    "gpu_sparse_layout": (
        PENDING,
        "consolidation_K10",
        "Reached only from gpu_sparse; same unblocker.",
    ),
    "external_memory": (
        PENDING,
        "consolidation_K10",
        "Streaming Dataset construction over sequence.mojo. Parked with "
        "sequence until an export and a chunk binding connect them together.",
    ),
```
(alternate_boosting, boosting_dart, boosting_rf keep their connect_17
entries; initialization, raw_data, model_dump are reachable and need none.)
