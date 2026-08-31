# Factorized verifier replay — scorecard

Generated for arm `factorized_v1` · contract `factorized_v1` · prompt `terra_factorized_review_v1` · labels v1.1 · integrity **OK**

Noise floor: Terra replica flips ran 6.4% (16/250) and humans matched run_1/run_2 8:6 — read every factorized-vs-stored delta against the replica column, never against zero.

## 1. Inputs and coverage

| input | sha256 | bytes |
|---|---|---|
| `C:\Users\Steven\PycharmProjects\realtorvision-backend\reports\labels_v1_1.json` | `7f9b03017195` | 99,808 |
| `artifacts_canary\factorized_v1_20260831\manifest.json` | `89f8fccdf276` | 3,091 |

Conditions answered: **1,047** · unit records {'redecided': 148} · labeled canary cards joined **88/88**

Production label cards excluded (by design): **37**. Canary-only is the scoreable population.

## 2. Full-population sweep (label-free)

All **1,047** replayed conditions. Stored Terra on the same population: `supported` 84.2% · `unsupported` 13.0% · `cannot_assess` 2.8%.

| derived class | n | share |
|---|---|---|
| `exact_and_warranted` | 589 | 56.3% |
| `misnamed_but_warranted` | 43 | 4.1% |
| `exact_but_trivial` | 98 | 9.4% |
| `misnamed_and_trivial` | 5 | 0.5% |
| `absent` | 149 | 14.2% |
| `inconclusive` | 163 | 15.6% |

### Degeneracy and health checks (G4)

| check | value | rule | verdict |
|---|---|---|---|
| `exact_and_warranted_share` | 56.3% | <= 90.0% | PASS |
| `misnamed_share` | 4.6% | >= 3.0% | PASS |
| `unclear_visible` | 1.3% | <= 10.0% | PASS |
| `unclear_claim_accurate_as_written` | 28.9% | <= 10.0% | **FAIL** |
| `unclear_material_enough_for_work` | 16.8% | <= 10.0% | **FAIL** |

*Evidence:* G1 and G2 are both satisfied by a verifier that answers `yes` to everything, and the category that would catch that (G3) is underpowered until L2. These label-free checks are what make the decisive gates trustworthy. No action is prescribed here.

## 3. Per-axis agreement with the human labels

### `visible` — 75/88 (85.2%)

| label | no | unclear | yes |
|---|---|---|---|
| `absent` | 1 | 0 | 2 |
| `exact` | 9 | 1 | 68 |
| `misnamed` | 1 | 0 | 6 |

### `claim_accurate_as_written` — 61/85 (71.8%)

| label | no | unclear | yes |
|---|---|---|---|
| `absent` | 0 | 1 | 2 |
| `exact` | 3 | 16 | 59 |
| `misnamed` | 2 | 2 | 3 |

### `material_enough_for_work` — 55/85 (64.7%)

| label | no | unclear | yes |
|---|---|---|---|
| `none` | 0 | 1 | 2 |
| `trivial` | 0 | 0 | 5 |
| `warranted` | 14 | 11 | 55 |

## 4. Class confusion (label → derived)

| label class | `exact_and_warranted` | `misnamed_but_warranted` | `exact_but_trivial` | `misnamed_and_trivial` | `absent` | `inconclusive` |
|---|---|---|---|---|---|---|
| `absent` | 2 | 0 | 0 | 0 | 1 | 0 |
| `exact_and_warranted` | 42 | 3 | 13 | 0 | 9 | 6 |
| `exact_but_trivial` | 4 | 0 | 0 | 0 | 0 | 1 |
| `misnamed_but_warranted` | 2 | 2 | 1 | 0 | 1 | 1 |

## 5. Gates

| gate | population | judged | fired | rule | verdict |
|---|---|---|---|---|---|
| **G1** false suppression | `supported_billed` n=57 | 57 | 10 | <= 5 · derived class suppresses work | **FAIL** |
| **G2** recovery of Terra's false rejections | `dirB_recovery` n=16 | 16 | 6 | >= 11 · visible == yes | **FAIL** |
| **G3a** misnamed catch | `misnamed_billed` n=7 | 7 | 2 | >= 4 · claim_accurate_as_written == no | **FAIL** (exploratory) |
| **G3b** trivial catch | `trivial_billed` n=5 | 5 | 0 | >= 3 · material_enough_for_work == no | **FAIL** (exploratory) |
| **G3c** absent catch | `hard_false_billed` n=3 | 3 | 1 | >= 2 · visible == no | **FAIL** (directional) |

Decisive result (G1 + G2 + G4, with G3 directional): **FAIL**.

*Evidence:* thresholds were pre-committed in `docs/HANDOFF_factorized_verifier_replay.md` §7 before the run. A decisive failure means the design gets revised, not the threshold. No action is prescribed here.

## 6. Disagreement audits

### Terra supported, factorized absent — one side is hallucinating (41)

- `redfin_10806500` `oc1_7d70c51ae086a11b` stored=supported derived=absent
- `redfin_10952874` `oc1_0807db3d5e597991` stored=supported derived=absent
- `redfin_10952874` `oc1_63292c2bf411d389` stored=supported derived=absent
- `redfin_10952874` `oc1_74eaab9cd204a68b` stored=supported derived=absent
- `redfin_11000447` `oc1_36d8fa941a803d52` stored=supported derived=absent
- `redfin_11000447` `oc1_e5469dc550971856` stored=supported derived=absent
- `redfin_11077450` `oc1_89ad6ab89688e3d0` stored=supported derived=absent
- `redfin_11079485` `oc1_882bfbf383946f88` stored=supported derived=absent
- `redfin_11185681` `oc1_1b4fee27685b292a` stored=supported derived=absent
- `redfin_11185681` `oc1_632bb9339aa62c29` stored=supported derived=absent
- `redfin_11185681` `oc1_ff239f3d892cf7d7` stored=supported derived=absent
- `redfin_125779232` `oc1_1e0e7b032023c8cc` stored=supported derived=absent
- `redfin_125779232` `oc1_3d1a0d63fb088044` stored=supported derived=absent
- `redfin_125779232` `oc1_7369dbd37bee24fd` stored=supported derived=absent
- `redfin_125779232` `oc1_8cbdae2f9bb3f32a` stored=supported derived=absent
- `redfin_125970550` `oc1_7523eef3a6530618` stored=supported derived=absent
- `redfin_125970550` `oc1_83c3115cf9901e78` stored=supported derived=absent
- `redfin_126224899` `oc1_63c3bbcd36bb9ab4` stored=supported derived=absent
- `redfin_126224899` `oc1_ba45c5daaa4d8976` stored=supported derived=absent
- `redfin_126224899` `oc1_e26a6139d357929b` stored=supported derived=absent
- `redfin_126418713` `oc1_10549b84d801b895` stored=supported derived=absent
- `redfin_126418713` `oc1_ba42ccbb1d813f8b` stored=supported derived=absent
- `redfin_127468088` `oc1_04d2943d36e6b9c2` stored=supported derived=absent
- `redfin_127468088` `oc1_89bd165f94cc0cd5` stored=supported derived=absent
- `redfin_127468088` `oc1_976e7ea07df8f23d` stored=supported derived=absent
- `redfin_166147710` `oc1_3a337d2373f8a0a5` stored=supported derived=absent
- `redfin_166147710` `oc1_51d732cb5528153f` stored=supported derived=absent
- `redfin_166147710` `oc1_83d04b2a94407233` stored=supported derived=absent
- `redfin_166147710` `oc1_8cddfc0a63490297` stored=supported derived=absent
- `redfin_166147710` `oc1_ea47eedfb3b43b31` stored=supported derived=absent
- `redfin_166147710` `oc1_f5ce61c6410fa96e` stored=supported derived=absent
- `redfin_25809814` `oc1_421a504b7114063e` stored=supported derived=absent
- `redfin_80877597` `oc1_874eea526d4e2aab` stored=supported derived=absent
- `redfin_80925528` `oc1_463b9e4badebe6a2` stored=supported derived=absent
- `redfin_80925528` `oc1_bdf60e6067e8b909` stored=supported derived=absent
- `redfin_80990371` `oc1_4107ebc052757d12` stored=supported derived=absent
- `redfin_80990371` `oc1_bd5632567e6d5300` stored=supported derived=absent
- `redfin_80990371` `oc1_df199e226e8c8bd7` stored=supported derived=absent
- `redfin_80990371` `oc1_e9ed24991a5bb5cb` stored=supported derived=absent
- `redfin_80990371` `oc1_ff31806fd5d908df` stored=supported derived=absent
- … 1 more (see the json)

### Terra rejected, factorized sees it — the recovery class (43)

- `redfin_10803207` `oc1_084ff89ca4edc1c4` stored=cannot_assess derived=inconclusive
- `redfin_10803207` `oc1_75c70a45e8012f91` stored=unsupported derived=exact_and_warranted
- `redfin_10803207` `oc1_bd04d765dac55f9b` stored=unsupported derived=exact_and_warranted
- `redfin_10806500` `oc1_32bf8869e625ef1f` stored=unsupported derived=exact_but_trivial
- `redfin_10952874` `oc1_42608fe1098b2fd3` stored=unsupported derived=exact_but_trivial
- `redfin_10952874` `oc1_4fb1ea73e633e933` stored=unsupported derived=exact_but_trivial
- `redfin_10952874` `oc1_8302615f64b86ad8` stored=unsupported derived=exact_and_warranted
- `redfin_10952874` `oc1_864108a3c69987e5` stored=unsupported derived=inconclusive
- `redfin_10952874` `oc1_b10d0b1346b9e17a` stored=unsupported derived=exact_but_trivial
- `redfin_10952874` `oc1_f9479bf4cec50774` stored=unsupported derived=inconclusive
- `redfin_11000447` `oc1_0e1dd7b103a3668e` stored=unsupported derived=inconclusive
- `redfin_11077450` `oc1_1a2fda15544c6ef5` stored=unsupported derived=exact_and_warranted
- `redfin_11077450` `oc1_261de865e0dd299d` stored=unsupported derived=exact_but_trivial
- `redfin_11077450` `oc1_d080389e1f69a331` stored=unsupported derived=inconclusive
- `redfin_11185681` `oc1_347ca41214e79635` stored=unsupported derived=exact_and_warranted
- `redfin_11185681` `oc1_425d26fff888e4d0` stored=unsupported derived=inconclusive
- `redfin_11185681` `oc1_447f84bf58498da7` stored=unsupported derived=exact_and_warranted
- `redfin_11185681` `oc1_8af6e2681d935d03` stored=unsupported derived=exact_and_warranted
- `redfin_125779232` `oc1_517203b47860a7b5` stored=unsupported derived=exact_and_warranted
- `redfin_125779232` `oc1_9cec506b34da7c09` stored=cannot_assess derived=inconclusive
- `redfin_126224899` `oc1_57552de0acadd8f2` stored=unsupported derived=exact_and_warranted
- `redfin_126418713` `oc1_3076cec157f5028b` stored=unsupported derived=exact_and_warranted
- `redfin_126418713` `oc1_da003b161d312a1e` stored=unsupported derived=inconclusive
- `redfin_127468088` `oc1_0636246f917808d6` stored=cannot_assess derived=inconclusive
- `redfin_127468088` `oc1_b2512c3ca661799e` stored=cannot_assess derived=inconclusive
- `redfin_127468088` `oc1_ce4012ad532858ac` stored=unsupported derived=exact_and_warranted
- `redfin_166147710` `oc1_6d404e73b5011de5` stored=unsupported derived=exact_and_warranted
- `redfin_166147710` `oc1_fccbc233d1bfed8b` stored=unsupported derived=exact_and_warranted
- `redfin_25809814` `oc1_d2af12342ab4be23` stored=unsupported derived=inconclusive
- `redfin_25809814` `oc1_eac5f9ac3822499a` stored=unsupported derived=misnamed_and_trivial
- `redfin_80877597` `oc1_f5459a0921a1c8b3` stored=unsupported derived=exact_and_warranted
- `redfin_80925528` `oc1_223827b97305711a` stored=cannot_assess derived=exact_but_trivial
- `redfin_80925528` `oc1_37259140a0a25612` stored=cannot_assess derived=exact_and_warranted
- `redfin_80925528` `oc1_511055d138621ad9` stored=unsupported derived=exact_and_warranted
- `redfin_80925528` `oc1_5e97f1da622ba53b` stored=unsupported derived=exact_and_warranted
- `redfin_80990371` `oc1_2ea4e1c604959e70` stored=unsupported derived=exact_but_trivial
- `redfin_80990371` `oc1_6ea2a14d497dece9` stored=unsupported derived=exact_and_warranted
- `redfin_80990371` `oc1_8d844512b3e5a7d5` stored=unsupported derived=exact_and_warranted
- `redfin_80990371` `oc1_c6c4c4017b1ed9dd` stored=unsupported derived=misnamed_but_warranted
- `redfin_80990371` `oc1_d7af7b65b4a20624` stored=unsupported derived=exact_but_trivial
- … 3 more (see the json)

### Factorized says misnamed — the future remap lane's input (50)

- `redfin_10806500` `oc1_a60baea9299f9b1f` stored=supported derived=misnamed_and_trivial — sees: masonry wall discoloration and patching
- `redfin_10806500` `oc1_f4c2c32f1c6dba1d` stored=supported derived=misnamed_but_warranted — sees: Swollen, peeling, deteriorated vanity cabinet surfaces.
- `redfin_10952874` `oc1_3c1a032f6a249ccb` stored=supported derived=misnamed_but_warranted — sees: weathered fence with leaning sections
- `redfin_10952874` `oc1_6506e0627a3977ec` stored=supported derived=misnamed_but_warranted — sees: moderate tub discoloration with aged caulk and dark grout
- `redfin_10952874` `oc1_7be7d3fe909651c6` stored=supported derived=misnamed_but_warranted — sees: areas of missing and scraped paint
- `redfin_10952874` `oc1_8178b1b9c823f719` stored=supported derived=misnamed_but_warranted — sees: Paint splatters, debris, and discoloration on hard flooring.
- `redfin_11000447` `oc1_09c51c2555f3b64f` stored=supported derived=misnamed_but_warranted — sees: Open seams and raised plank edges in the worn flooring.
- `redfin_11000447` `oc1_4f0058d96b06f3a3` stored=supported derived=misnamed_but_warranted — sees: The vanity base/toe-kick is detached, exposing the lower interior.
- `redfin_11000447` `oc1_961e6edcac992934` stored=supported derived=misnamed_but_warranted — sees: Patterned flooring with visible discoloration and debris.
- `redfin_11000447` `oc1_be4b77f1d023f9bd` stored=supported derived=misnamed_but_warranted — sees: Worn and soiled older wood plank flooring.
- `redfin_11000447` `oc1_fa37e7520234829c` stored=supported derived=misnamed_but_warranted — sees: Open seams and raised plank edges in addition to worn finish.
- `redfin_11077450` `oc1_f5a8a54a0117c5b3` stored=supported derived=misnamed_but_warranted — sees: large graffiti and uneven paint patches
- `redfin_11185681` `oc1_1c17f0199476323d` stored=supported derived=misnamed_but_warranted — sees: older tan square tile with darkened grout
- `redfin_11185681` `oc1_5286f255058331d4` stored=supported derived=misnamed_and_trivial — sees: water spotting on shower glass
- `redfin_11185681` `oc1_69e1058725f35008` stored=supported derived=misnamed_but_warranted — sees: chipped paint and discoloration on the lower walls
- `redfin_125779232` `oc1_56be9b9975981f37` stored=supported derived=misnamed_but_warranted — sees: dirty, scuffed parquet-pattern flooring
- `redfin_125779232` `oc1_777cd635cf27aeee` stored=supported derived=misnamed_but_warranted — sees: worn and discolored parquet-pattern flooring
- `redfin_125970550` `oc1_562dd1da5b0d9252` stored=supported derived=misnamed_but_warranted — sees: Corroded fixtures with damaged or missing shower-valve trim.
- `redfin_125970550` `oc1_7ec3de5f129a5fd5` stored=supported derived=misnamed_but_warranted — sees: heavily soiled white tile flooring
- `redfin_125970550` `oc1_8bf0920221ea6796` stored=supported derived=misnamed_but_warranted — sees: water staining and torn wall paneling
- `redfin_125970550` `oc1_9ec347a23f3ed720` stored=supported derived=misnamed_but_warranted — sees: Stained tile flooring with cracked and missing pieces.
- `redfin_125970550` `oc1_dd9ca495af2f633a` stored=supported derived=misnamed_but_warranted — sees: torn and lifted sheet flooring
- `redfin_126224899` `oc1_17e2478acfc349e8` stored=supported derived=misnamed_but_warranted — sees: worn and stained resilient flooring
- `redfin_126224899` `oc1_4900757ed55e17c9` stored=supported derived=misnamed_but_warranted — sees: dark lower-wall staining and finish deterioration
- `redfin_126224899` `oc1_4f7604e7064dce56` stored=supported derived=misnamed_but_warranted — sees: dark ceiling staining near the window corner
- `redfin_126224899` `oc1_8b6bf7c321af5b5d` stored=supported derived=misnamed_but_warranted — sees: heavy dark staining and growth-like discoloration on lower walls
- `redfin_126224899` `oc1_999efd853008d114` stored=supported derived=misnamed_and_trivial — sees: grime and discoloration on vanity surfaces
- `redfin_126224899` `oc1_a72e30e0a8cb65b6` stored=supported derived=misnamed_but_warranted — sees: weathered fence with leaning sections
- `redfin_126224899` `oc1_d32b38b4d91549c0` stored=supported derived=misnamed_and_trivial — sees: heavy surface grime and dark speckling on the vanity countertop
- `redfin_126224899` `oc1_efe915a40d1bb026` stored=supported derived=misnamed_but_warranted — sees: soiled and worn carpet
- `redfin_126418713` `oc1_e6be634a3ab8e0f6` stored=supported derived=misnamed_but_warranted — sees: heavy swirled plaster ceiling texture
- `redfin_127468088` `oc1_6a1409628ae9f4d7` stored=supported derived=inconclusive — sees: dark staining and discoloration at the rear wall and lower area
- `redfin_127468088` `oc1_7d6a7e398269765b` stored=supported derived=misnamed_but_warranted — sees: Mottled, flattened carpet with visible discoloration.
- `redfin_127468088` `oc1_a4bd1f8bffa6aaf2` stored=supported derived=misnamed_but_warranted — sees: worn, discolored carpet
- `redfin_127468088` `oc1_f2c05ba24db79c17` stored=supported derived=misnamed_but_warranted — sees: patched drywall and cracking near the outlet
- `redfin_166147710` `oc1_5a69289015183dbc` stored=supported derived=misnamed_but_warranted — sees: Open rectangular cutouts and removed mounting strips in painted wall paneling.
- `redfin_166147710` `oc1_da54345fcbb9a21e` stored=supported derived=misnamed_but_warranted — sees: widespread wall and ceiling staining and discoloration
- `redfin_25809814` `oc1_eac5f9ac3822499a` stored=unsupported derived=misnamed_and_trivial — sees: white horizontal blinds
- `redfin_80925528` `oc1_5f38b2b4122714d0` stored=supported derived=misnamed_but_warranted — sees: surface dirt and scattered dark marks on the green flooring
- `redfin_80925528` `oc1_c069fa037c3edff7` stored=supported derived=misnamed_but_warranted — sees: wall cracks and missing wall material
- … 10 more (see the json)

## 7. Stability benchmark

Both rates below are measured on the 702 conditions the two replicas share, keyed by (property, catalog item, unit) — condition_id embeds the run id, so replicas share none of those.

- Factorized vs stored run_1: **39/702** (5.6%) on Terra's presence axis
- Stored run_1 vs run_2 (the noise floor): **42/702** (6.0%)
- Factorized vs run_1 across all replayed conditions: **101/1047**

## 8. Prompt adherence

- Misnamed answers with no `observed_description`: **0**
- `visible = no` answers whose other factors are not `unclear`: **0**

## 9. What this does not show

- Production listings. This scores the canary only (see §inputs for the excluded count): production contributes one misnamed and one trivial card, not enough to pay for a second input path.
- A production error rate. The labeled cards were sampled disagreements-first, so catch rates sit on a hard, non-representative slice, and the full-sweep class distribution is not a population rate.
- Specificity on Terra's correct rejections. There is no dirB loss population (`dirB_terra_correct` = 0 cards), so G2 measures recovery with no paired over-revival check until Session L2 adds one.
- Anything downstream: Sol, packages, pricing, dispositions, or whether a trivial condition should be absorbed by turnover rather than dropped.
