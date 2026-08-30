# Why the v10 JEPA-WM adapter failed on retained retreat

Research date: 2026-08-29

## Short answer

The failure does **not** show that JEPA cannot understand negative motion. It
shows that this adaptation setup did not make the signed effect of the retreat
action sufficiently identifiable and influential inside the frozen world
model.

The authenticated held-out pattern is unusually clean: the recorded action
lost to zero action on all 50 attachment/retreat contexts in each seed, then
won all 118 alignment/insertion/hold contexts. A generic precision problem
would not normally split exactly at a semantic phase boundary. The strongest
model-side interpretation is that the model can predict the later visual
transitions but its learned energy still treats the negative-X retained
retreat as less explanatory than persistence.

This note separates claims made by the papers and source code from inferences
about that measured result.

## What JEPA learns—and what it does not

The original JEPA idea is to predict a target **representation** from a context
representation instead of reconstructing pixels. I-JEPA and V-JEPA train this
feature-prediction objective without action labels; V-JEPA specifically uses
feature prediction without pixel reconstruction, text, or human labels.
Consequently, a visual JEPA representation can encode useful appearance and
motion information without attaching any intrinsic meaning to a robot command
such as “negative X.”

Primary sources:

- [I-JEPA paper](https://arxiv.org/html/2301.08243)
- [V-JEPA paper](https://arxiv.org/abs/2404.08471)
- [V-JEPA 2 paper, representation-space pretraining objective](https://arxiv.org/html/2506.09985#S2.SS1)

Action conditioning is a subsequent learning problem. V-JEPA 2 explicitly
states that action-free pretraining does not account for the effect of an
agent's actions. V-JEPA 2-AC therefore freezes the video encoder and trains a
new, roughly 300-million-parameter predictor from robot trajectories. That
predictor consumes visual features, end-effector state, and action tokens, and
is trained with next-feature teacher forcing plus a two-step autoregressive
rollout loss. At planning time, an action sequence has low energy when its
predicted terminal feature is close to the encoded goal feature; the model
executes only the first action and replans.

- [V-JEPA 2-AC training objective and architecture](https://arxiv.org/html/2506.09985#S3.SS1)
- [V-JEPA 2-AC goal-energy planning](https://arxiv.org/html/2506.09985#S3.SS2)
- [Official V-JEPA 2 action-conditioned predictor](https://github.com/facebookresearch/vjepa2/blob/main/src/models/ac_predictor.py)

The upstream JEPA-WM paper generalizes the same recipe: a frozen visual
encoder maps observations into features, an action encoder maps commands into
features, and a trained predictor forecasts the next state embedding. The
training loss is feature prediction on observed state-action trajectories;
planning minimizes distance between an unrolled terminal feature and a goal
feature. The paper is explicit that this recipe has no policy/value/reward
head and offers no theoretical guarantee that an optimized plan is feasible.

- [JEPA-WM definition, training, and planning equations](https://arxiv.org/html/2512.24497v3#S3)
- [JEPA-WM scope and lack of feasibility guarantees](https://arxiv.org/html/2512.24497v3#S1)

Thus neither “good visual representation” nor “low average prediction loss”
implies that every signed robot coordinate has been grounded correctly. A
command axis is an embodiment convention learned from action-conditioned
trajectories, not a semantic property supplied by JEPA pretraining.

## What Quantis actually trained

Quantis is based on the official DROID JEPA-WM with a frozen DINOv3 ViT-L/16
visual encoder and a frozen 12-layer AdaLN predictor. The official checkpoint
was trained on DROID and is loaded using its DROID normalization and evaluation
configuration ([`model.py`](../../jepa_wm/model.py)).

The v10 run was much more constrained than upstream world-model training:

1. Every parameter is frozen.
2. Only `model.model.predictor.action_encoder.weight` is made trainable.
3. The action encoder's bias, visual encoder, and entire nonlinear predictor
   remain fixed.

This is implemented in [`adapter.py`](../../jepa_wm/adapter.py) and
[`adapt_recording.py`](../../jepa_wm/adapt_recording.py). The trainable object
is a 7,168-parameter global linear map from the seven DROID action coordinates
into the predictor's existing action-token space. It can rotate and rescale
how actions enter the old DROID predictor. It cannot:

- make a visually weak connector displacement newly salient to DINOv3;
- create a new state-dependent or phase-dependent action interaction;
- teach the frozen predictor new rigid-attachment/contact dynamics; or
- change the latent geometry used as the terminal goal cost.

The local objective is stronger than the vanilla JEPA-WM prediction loss. For
each demonstrated rollout it minimizes recorded-action terminal L2 energy and
adds margins against zero action, a different-context action, and the
currently most deceptive sampled local candidate. The exact loss is visible
in [`adapt_recording.py`](../../jepa_wm/adapt_recording.py), and energy is the
distance between the action-unrolled terminal latent and the recorded target
latent in [`rollout_scoring.py`](../../jepa_wm/rollout_scoring.py).

Therefore it would be inaccurate to say that our training never told the model
that recorded retreat should beat zero. It did. The result says the one shared
linear weight could not preserve that ranking for retreat while fitting the
rest of the corpus under this optimization budget.

## Paper-backed limitations directly relevant here

### 1. Frozen visual features are not metric robot state

The JEPA-WM study reports that frozen visual embeddings preserve appearance
and coarse spatial layout, while precise joint/end-effector quantities are
only implicit and patch-quantized. It finds proprioception especially helpful
near goals, where small physical displacement can produce negligible feature
distance. It also finds DINO's fine object segmentation important because
localized object motion becomes easier for a predictor to learn.

- [JEPA-WM: impact of proprioception and small displacements](https://arxiv.org/html/2512.24497v3#S5.SS2)
- [JEPA-WM: frozen encoder and object-local dynamics](https://arxiv.org/html/2512.24497v3#S5.SS2)

This means 2 mm in task space is not a fixed amount in latent space. A retreat
and an alignment of comparable Cartesian magnitude may move different image
patches, cross different occlusions, or change connector/socket relationships
by very different latent distances.

### 2. How action information enters the predictor matters

The JEPA-WM study treats action conditioning as a first-order architectural
choice. Its preferred AdaLN design injects action information at every
transformer block; the authors argue that this can prevent action information
from weakening through depth. They also find task-dependent differences among
conditioning schemes even when action capacity is controlled, and note that
equal rollout losses can coexist with different planning behavior.

- [JEPA-WM action-conditioning architectures](https://arxiv.org/html/2512.24497v3#S11.SS2)
- [JEPA-WM equalized action-ratio results](https://arxiv.org/html/2512.24497v3#S15.SS1)

The official DROID config used here does have the stronger AdaLN predictor,
but our adapter does not retrain that conditioning path. It changes only the
linear action embedding feeding the already-trained path.

### 3. Accurate rollout prediction is not sufficient for planning

The JEPA-WM paper warns that faithful action unrolling does not immediately
imply successful planning. It reports model failures involving biased spatial
predictions and hallucinated grasping, and shows that planning performance is
sensitive to representation, action conditioning, optimizer, and data
diversity. V-JEPA 2-AC separately reports camera-axis ambiguity: with an
uncalibrated monocular view, the robot coordinate frame may not be visually
identifiable, leading to world-model error.

- [JEPA-WM: rollout fidelity is not sufficient](https://arxiv.org/html/2512.24497v3#S5.SS2)
- [JEPA-WM manipulation failure examples](https://arxiv.org/html/2512.24497v3#S15.SS1)
- [V-JEPA 2-AC camera-coordinate limitation](https://arxiv.org/html/2506.09985#S4.SS3)

## Inference from the Quantis evidence

The following points are diagnoses, not claims proved by the papers.

### The retreat action is insufficiently identifiable

Every training state has one observed future produced by one scripted action.
Our contrastive terms label zero and sampled commands as worse, but they do not
provide the *actual alternative futures* from the same physical state under
zero, positive-X, and negative-X interventions. As a result, training can rank
examples without learning a calibrated counterfactual transition function.
Phase appearance and deterministic trajectory progress can remain strong
shortcuts.

The clean 0/50-versus-118/118 phase split is consistent with a shortcut: for
the repetitive retained retreat, a persistence prediction can remain closer
to the target latent than the predictor's miscalibrated negative-X response.
Alignment and insertion can still pass if their visual transitions are more
distinctive or already better represented by the frozen DROID dynamics.

### One global map has conflicting jobs

The adapter must use the same action projection for attachment, retreat,
alignment, insertion, and hold. Its epoch contains 50 retreat versus 118
post-retreat examples per held-out-equivalent trajectory. Even with equal
sample visitation, gradients need not have equal size, and updates that improve
the larger/louder set can undo a margin learned for the visually quieter set.
The exact final score—118/168 on both held-out seeds—is consistent with the
optimizer preserving every post-retreat ranking while sacrificing the entire
retreat regime.

That is evidence of a capacity/compatibility boundary in this adapter, not
proof that another epoch alone would solve it.

### “Negative X” is a symptom, not the semantic cause

The same physical magnitude in positive and negative directions need not be
symmetric after camera projection, DINOv3 encoding, action normalization, and
the frozen nonlinear predictor. The proper conclusion is therefore:

> The frozen representation and globally linear adaptation contract did not
> make the signed retained-retreat effect sufficiently identifiable and
> salient.

It is too strong to conclude that JEPA generally cannot model negative-X
motion, or that the recording/controller commanded the wrong direction.

## What the current literature suggests changing

This section extends the diagnosis with primary literature available through
2026-08-29. The August 2026 results below are recent preprints, so they are
useful architectural evidence rather than settled prescriptions for Quantis.

### Paper-backed directions

**Keep local visual structure visible to the cost.** DINO-WM predicts spatial
DINO patch features rather than first compressing the image to one global
state. The JEPA-WM study likewise attributes part of DINO's manipulation
advantage to localized object features. V-JEPA 2.1 makes dense spatial and
temporal grounding an explicit pretraining goal: visible and masked tokens both
contribute to its dense predictive loss, and multiple intermediate encoder
layers receive self-supervision. DDP-WM addresses the dynamics side by
localizing sparse primary dynamics caused by physical interaction separately
from context-driven background updates. A separate 2026 study shows that a
small learned control-relevant projection can sit on top of a frozen visual
encoder, although its experiments target background invariance rather than
submillimetre manipulation.

- [DINO-WM spatial patch prediction](https://arxiv.org/abs/2411.04983)
- [JEPA-WM localized dynamics finding](https://arxiv.org/html/2512.24497v3#S5.SS2)
- [V-JEPA 2.1 dense predictive loss](https://arxiv.org/abs/2603.14482)
- [DDP-WM primary-dynamics localization](https://arxiv.org/abs/2602.01780)
- [Bisimulation projection over frozen visual features](https://arxiv.org/abs/2602.18639)

**Ground the geometry used for planning, not only a decoder.** PSG-JEPA adds
training-only heads for robot state and multi-horizon state change and reports
better probing, planning, and policy results without inference-time overhead.
SCALE makes the sharper metric point: state may be decodable yet contribute
little to Euclidean candidate ranking. Its pairwise state-distance calibration
improves every reported task-solver average, while a state-regression control
with comparable decodability is less consistent. PhyLatent independently
reports physical-identifiability and counterfactual-action collapse and adds
state grounding plus explicit separation of action branches.

- [PSG-JEPA physical state and transition grounding](https://arxiv.org/html/2608.06799)
- [SCALE state-calibrated planning geometry](https://arxiv.org/html/2608.16287)
- [PhyLatent counterfactual branch separation](https://arxiv.org/html/2608.05720)

**Use state where vision is metrically weak.** The JEPA-WM ablations find that
proprioception consistently helps near goals, where small physical changes can
be nearly invisible to a frozen feature distance. V-JEPA 2-AC goes much
further than Quantis: it freezes the vision encoder but trains an approximately
300-million-parameter predictor conditioned on visual tokens, end-effector
state, and actions.

- [JEPA-WM proprioception ablation](https://arxiv.org/html/2512.24497v3#S5.SS2)
- [V-JEPA 2-AC architecture](https://arxiv.org/html/2506.09985#S3.SS1)

**Represent contact regimes explicitly when they change the dynamics.** Work
on hybrid dynamics shows that one monolithic transition function can average
across contact modes. Established nested-mixture work injects contact and
kinematic priors into specialized experts; the newer PRISM-WM uses learned
context gating over mode-specialized transition experts. Real-world
visuo-tactile servoing also succeeds with explicit binary contact, contact
line, and wrench state rather than asking vision alone to infer contact.

- [Nested mixture of experts for hybrid dynamics](https://proceedings.mlr.press/v144/ahn21a.html)
- [PRISM-WM context-gated hybrid dynamics](https://arxiv.org/html/2512.08411)
- [Explicit contact features for contact servoing](https://proceedings.mlr.press/v205/merwe23a.html)

None of these papers proves that a mixture of experts, a particular auxiliary
loss, or tactile sensing will fix this connector task. They establish narrower
facts: local features matter; planner-facing geometry can hide decodable state;
physical and transition grounding can help; and contact systems benefit from
mode-dependent structure. In particular, DDP-WM supports *spatially separating
interaction-driven motion from background updates*, not splitting dynamics by
our scripted phases. The NMOE and PRISM-WM results directly support specialized
dynamics for hybrid/contact regimes in other systems, but the proposed
retreat-versus-align/insert gate is our inference from the exact Quantis phase
boundary, not a result those papers tested.

### Architectural inference for Quantis

Quantis already retains the DINO token tensor, but
[`terminal_l2_energy`](../../jepa_wm/objective.py) uniformly averages squared
error over every spatial and feature dimension. Connector/socket motion can
therefore be diluted by the unchanged image. It also has authoritative phase
and attachment labels in [`insertion_layout.py`](../../jepa_wm/insertion_layout.py),
yet its one trainable global action matrix cannot use either label.

A bounded capacity ladder follows from those facts:

1. **Measure before changing capacity.** Plot per-token and connector/socket
   region energy, plus negative-X/zero/positive-X curves, by phase. A pose probe
   is not enough: also measure whether latent distance ranks task-space pose
   and connector/socket distances correctly.
2. **Keep DINOv3 frozen, but learn a small planning projection/cost.** Train it
   with standardized end-effector and connector-to-socket pose distances plus
   multi-horizon state-change targets. Score a weighted global-and-local
   objective so both scene consistency and fine geometry survive. This adapts
   the lesson from SCALE/PSG-JEPA; their exact implementations are not proven
   drop-in retrofits for this frozen checkpoint.
3. **Add known-regime action capacity before a learned MoE.** Use the existing
   phase/attachment state to gate small low-rank action residuals or per-layer
   modulation around a shared base. That directly tests whether retreat and
   align/insert gradients conflict without the data and routing risk of a full
   learned expert system. It is a Quantis design inference, not a paper result.
4. **Collect paired counterfactuals only if the offline gate still demands
   them.** Same-reset negative-X, zero, and positive-X futures would supply the
   causal branch separation that observational ranking negatives cannot. This
   is the closest local analogue of PhyLatent's counterfactual constraint.
5. **Unfreeze predictor blocks only after those ablations.** The upstream
   systems train the action encoder and nonlinear predictor; V-JEPA 2-AC trains
   hundreds of millions of predictor parameters. Jumping directly there would
   add far more capacity and regression surface than the measured failure yet
   justifies.

Force should supplement, not replace, phase/attachment state: the failed
retreat contexts have retained attachment but zero connector force. Any signal
used at inference must also be observable at the claimed deployment boundary;
simulator-only geometry may supervise training, as in the papers, but cannot be
silently required by an image-only runtime claim.

### Comparison with JEPA-WM and V-JEPA 2-AC

| System | Frozen | Learned action/dynamics path | State and planner geometry |
| --- | --- | --- | --- |
| V-JEPA 2.1 encoder | none during pretraining | not an action-conditioned robot dynamics model by itself | dense loss over visible and masked tokens plus deep self-supervision; improves real-robot grasping when used in the paper's pipeline |
| JEPA-WM recommended setup | visual encoder | action encoder and 12-layer AdaLN predictor | proprioception is optional but empirically helpful; spatial feature error |
| V-JEPA 2-AC | visual encoder | new ~300M action-conditioned predictor | explicit end-effector state and action tokens; terminal feature energy |
| Current Quantis v10 | visual encoder and nonlinear predictor | one 7,168-weight global linear action map | no explicit pose/phase/contact conditioning; terminal error uniformly averaged over tokens/features |
| Bounded Quantis proposal | visual encoder | shared base plus small known-regime residuals; unfreeze only if needed | learned task-calibrated global/local cost, state/change grounding, phase and attachment conditioning |

The proposal is therefore not “make JEPA bigger.” It keeps the expensive
pretrained perception frozen, but makes fine task geometry visible to the
planner and gives demonstrably different physical regimes limited separate
capacity.

## Fast tests that would discriminate the hypotheses

Before another full training run:

1. Sweep negative-X, zero, and positive-X actions at matched retreat and
   alignment contexts and plot terminal energy. This reveals whether the
   retreat energy surface has the wrong sign, is flat, or merely has a biased
   minimum.
2. Encode paired same-reset counterfactual clips for negative-X, zero, and
   positive-X probes without fitting. Measure whether DINOv3 target distances
   visibly separate those outcomes.
3. Fit a balanced retreat-only diagnostic adapter and test it against both
   retreat and post-retreat held-out contexts. If retreat becomes correct while
   later phases regress, the shared linear map is demonstrably conflicting.
4. Compare that diagnostic with a phase/attachment-conditioned action adapter
   or a narrowly unfrozen final predictor block. This tests whether additional
   state-dependent capacity—not more identical epochs—is required.
5. Add precise end-effector/attachment conditioning in an offline ablation.
   The JEPA-WM paper's proprioception result predicts the largest benefit where
   RGB latent differences are small.

These are offline diagnosis steps. They do not require weakening any force,
tracking, insertion, or held-out gate, and none by itself authorizes another
simulator action.
