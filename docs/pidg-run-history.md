# PIDG run labels and validation history

The 4.17× headline belongs to **R06**, whose recorded variant is `both`. This selects both module paths from the candidate tree; it does not mean that both selected components contain acceleration patches. R06 retains the R03 flow optimization and restores the original raster CUDA source.

## Checks in the published summary

All rows below contain 2,435 comparisons and use the same frozen thresholds. Indices refer to `results/pidg/summary.json`.

| Check | Run | Recorded variant | Effective configuration | Generic status | Bitwise identical | Continuous violations |
|---|---|---|---|---|---:|---:|
| `checks[0]` | R01 | `flow-only` | Dense embedding gather backward + original raster | `INCORRECT` | 1,168 | 82 |
| `checks[1]` | R03 | `flow-only` | Exact-zero-eliding gather backward + original raster | `PASS` | 1,360 | 0 |
| `checks[2]` | R04 combined | `both` | R03 gather + grouped raster gradient reduction | `INCORRECT` | 1,367 | 84 |
| `checks[3]` | R04 raster-only | `raster-only` | Original flow + grouped raster gradient reduction | `INCORRECT` | 1,372 | 84 |
| `checks[4]` | R06 retained | `both` | R03 gather + original raster source rebuilt in the candidate tree | `PASS` | 1,368 | 0 |

R01 replaced indexed gather with `F.embedding` and was rejected. R03 replaced that implementation with the retained gather backward that omits exactly-zero gradient vectors. This is a different implementation, not a rerun of the failed R01 candidate. R02 timing was configuration-invalid because intended CUDA architecture flags were omitted; the GLM include-path build fix precedes subsequent valid measurements.

R04 introduced a raster reduction patch. The raster-only and accelerated combined configurations each had 84 threshold violations and were rejected. Before R06, the raster CUDA source was restored byte-for-byte to baseline and the candidate-local extension rebuilt. The flow source stayed unchanged from R03. The historical `both` label was preserved because both selected module paths came from that restored candidate tree.

The final dry_ice confirmation and its paired numerical check are R06: `confirmation.variant == "both"`, `checks[4]`, **9369.79 → 2244.31 ms, 4.17×**, and **1,368** bitwise-identical checks. The earlier R03 flow-only PASS has **1,360**, not 1,368.

## Source and threshold provenance

| Source | SHA256 |
|---|---|
| R01 flow implementation | `7eb6e1f38c7c612396c6cc8246c89c3c0dcc5c87de5717282dcb03b6013c8dda` |
| Retained R03/R06 flow implementation | `3566f99af458574e465354834814671e38e00ce95212cca6782acdca8c26a37b` |
| Rejected R04 raster backward | `9edf9e28ff268f3f132ec7371e76c1c25bbca1b2f5640d8dc8fcea2227273f1a` |
| Original / restored R06 raster backward | `ea9d57d653e3938f9723112e592909c8e022df6d7a86eedfabde3185a481ac6b` |
| Rebuilt R06 raster binary | `08247b563cc885647cc164ad726bc81eb885f6058e723a529030064143fe3b48` |
| Frozen thresholds file, all five checks | `ea0ad940fe4795b23b2cb6f3277f1747304025b581b5eb2d42a51cd5cc8d8232` |

The machine-readable [run provenance](../results/pidg/run-provenance.json) reconciles the locally archived original check records, final optimizer receipt, source manifests and publication-validated baseline source. Historical labels, failures, raw checks and thresholds remain unchanged.

## Judge status

The R03/R06 generic module checks report `PASS`; this is separate from formal ResearchBoost acceptance. The published result explicitly records **`formal_ResearchBoost_acceptance: false`** and **`formal_benefit_verdict: UNAVAILABLE_SHARED_WORKLOAD`**. The 3.27% observed timing-resolution threshold is a module diagnostic, not a formal benefit verdict. No complete-training frozen-judge PASS is claimed.
