# Robustness testing strategy

The rest of the test suite asks whether a feature is **correct**. This document
is about whether it **holds up**: what Gala does when the input is a selection
string someone mistyped into a panel field, a structure with no atoms in it, a
coordinate that came back as `nan`, a grid whose header lies about its size, or
a `.pse` that is not a session.

`tests/test_robustness.py` (everything that is just data),
`tests/test_robustness_blender.py` (everything that needs a scene) and
`tests/test_robustness_structures.py` (an awkward structure),
`tests/test_robustness_lifecycle.py` (molecules coming and going),
`tests/test_robustness_modification.py` (a molecule edited after Gala met it)
and `tests/test_robustness_multi.py` (several at once) implement the permanent
part of what follows. The rest is a plan, so
that the cases we decided *not* to run on every commit are written down rather
than forgotten.

---

## 1. What counts as a robustness defect

Gala is a library with a text-field UI on top of it, embedded in an application
that shows a Python traceback in a console the user is probably not looking at.
So the bar is not "does not crash". It is one of three contracts:

| Contract | Means |
|---|---|
| **It raises the documented error** | A mistyped selection is a `SelectionSyntaxError` carrying the string and a caret, not a `TypeError` from three modules down. The operator layer turns Gala's own exceptions into `self.report({'ERROR'})`; anything else becomes a console traceback. |
| **It refuses rather than guesses** | A truncated grid, a short mask, a session written with `pse_binary_dump` — reported, never quietly padded, clamped or reinterpreted. A figure that is subtly wrong is worse than one that failed to build. |
| **It survives** | An empty structure, a Unicode residue name, a `nan` B-factor: an empty result and no traceback. |

A fourth, implicit: **it terminates**. Anything sized from a header field or a
user-supplied ratio can be asked for more memory or more iterations than exist.

---

## 2. The attack surface

Where untrusted or unusual input actually enters:

| Boundary | Modules | What arrives |
|---|---|---|
| **Selection language** | `core/selection.py` | Arbitrary user text, from the API and from panel fields. Tokeniser, recursive-descent parser, expression tree, KD-tree evaluation. The single largest surface. |
| **Structure adapter** | `core/entity.py`, `core/attributes.py` | Anything callers pass as "a structure": biotite arrays, MN molecules, Blender objects, or none of those. Atom counts that no longer match vertex counts. |
| **Session deserialisation** | `pymol/session.py` | A **pickle** — the only genuinely hostile input Gala reads. Allow-listed globals; every field index is a trust boundary. |
| **Session → scene** | `pymol/load.py`, `pymol/save.py` | Object names Blender must accept as datablock names, colour indices, group hierarchies, per-state coordinates. |
| **OpenDX grids** | `electrostatics/grid.py` | A text header that declares an allocation size, then a data block that may not match it. |
| **External processes** | `electrostatics/apbs.py` | `subprocess` to APBS and PDB2PQR: executable discovery, `PATH`/env overrides, exit codes, output files that may not appear. |
| **CSV per-residue data** | `color/coloring.py` | Spreadsheet exports: encodings, blank cells, decimal commas, duplicate keys. |
| **Numerical algorithms** | `core/geometry.py`, `measure/`, `interactions/detect.py`, `interactions/perception.py`, `electrostatics/grid.py` | Coincident atoms, collinear rings, zero-length vectors, non-finite coordinates, cutoffs the caller chose. |
| **Colour** | `color/colormaps.py`, `pymol/palette.py` | Hex strings, out-of-range indices, values outside `[0, 1]`, sRGB/linear round trips. |
| **Blender state** | `ops/operators.py`, `ui/`, `core/viewport.py` | Operators invoked with no active object, in the wrong mode, twice, or out of order. Edit Mode versus Object Mode is a genuine state machine. |
| **Filesystem** | `pymol/`, `electrostatics/`, `scene/render.py` | Paths that do not exist, are directories, are unwritable, or have a misleading suffix. |

**Not in scope:** Gala never opens a socket and never fetches a URL (Molecular
Nodes does the fetching). There is no concurrency in Gala itself — no threads,
no multiprocessing — so "concurrency" reduces to *re-entrancy* (an operator run
again before the first finished, a depsgraph evaluated mid-write) and to the
one real shared mutable: Molecular Nodes' session dictionary of entities.

---

## 3. The four tiers

### Tier 1 — permanent, in the normal suite

Runs on every commit, inside Blender, in well under a second. Criteria for
belonging here:

- deterministic and fast (no sleeps, no large allocations, no subprocesses);
- the input is one a user could plausibly produce, or one that has already been
  seen in the wild;
- the assertion is a contract, not an implementation detail.

Implemented in the three `test_robustness*.py` modules, organised by boundary.
Where the exact exception type is not the contract, the test accepts a tuple —
the point is that it *refuses*, not that it refuses with a particular class.

Most of these tests began as `xfail(strict=True)` records of defects found by
probing the code — each stating the contract that *should* hold and naming the
bare exception that escaped. That is the workflow to repeat: record the gap as
a strict xfail, fix the code, and let the resulting `XPASS(strict)` failure tell
you the fix landed before you delete the marker. See §5 for what was found this
way and what was done about it.

### Tier 2 — property-based and fuzz, run periodically

Not on every commit: they need `hypothesis` (not currently a dependency) or a
corpus, and they are non-deterministic by design. Suggested weekly, or on a
label, writing failures back into Tier 1 as fixed cases.

| Target | Property |
|---|---|
| `select` / `describe_selection` | **Round trip.** For any mask over any structure, `select(a, describe_selection(a, mask)) == mask`. This is the single highest-value property in the codebase — the function already self-verifies, so the fuzzer is testing the fallback logic. |
| Selection parser | **No unexpected exception class.** For any string, `select` raises `SelectionSyntaxError` or returns a bool mask of the right length. Nothing else. Grammar-aware generation (valid tokens in invalid orders) finds more than random bytes. |
| `expand_selection` | **Monotone and idempotent.** Growing never shrinks; expanding twice at the same level equals expanding once; a larger distance never yields a smaller mask. |
| Colormap sampling | `sample` is total on `[0, 1]`, monotone in position for a monotone map, and `srgb_to_linear`/`linear_to_srgb` round trip to within 1e-6 on `[0, 1]`. |
| `PotentialGrid.sample` | Trilinear interpolation reproduces node values exactly at nodes, and stays within the min/max of the eight surrounding corners everywhere else. |
| Session round trip | `read_session(write_session(s)) == s` for generated sessions: any atom count, any state count, arbitrary Unicode names, `nan` coordinates, colour indices at the ends of the table. |
| `.pse` reader | **Structured fuzzing.** Take `tests/data/session.pse`, unpickle it with the safe unpickler, mutate the tree (truncate lists, swap types, replace numbers with strings, `nan`, huge ints), re-pickle, read. Every result must be a `PymolSessionError` or a valid `PymolSession`. Byte-level mutation mostly produces unpickling errors and is much less productive. |
| OpenDX reader | Header/body mutation: counts and `items` disagreeing, missing deltas, non-numeric tokens mid-block, gzip truncation. |
| `safe_name` | Total, idempotent, never empty, never leading-dot, always a valid geometry-nodes attribute name. |

### Tier 3 — expensive integration and stress, on a schedule

Nightly or weekly; minutes, not milliseconds. These are the ones that need a
real Blender scene, a real solver, or a large structure.

- **Large structures.** A ribosome-scale import (~10⁵–10⁶ atoms) through
  `find_interactions`, `expand_selection`, `describe_selection`, `color_by_*`.
  Watch wall-clock *and* peak RSS: several paths are quadratic in the number of
  atoms matched, and the KD-tree ones allocate per-pair.
- **Adversarial geometry at scale.** Every atom at the same coordinate; every
  atom collinear; a structure that is one residue repeated 10⁵ times. These
  turn KD-tree queries into all-pairs work.
- **The APBS pipeline end to end**, including a solver that diverges (`inf` in
  the map), one that is killed mid-run, and a `GALA_APBS` pointing at something
  that is not an executable.
- **Repeated operations.** `publication_setup` a hundred times on one scene;
  `load_session` of the same file repeatedly; alias create/delete cycles.
  Assert on datablock counts, not just absence of exceptions — Blender leaks
  materials, node groups and images very willingly.
- **Render smoke tests** at tiny resolution across the presets, EEVEE and
  Cycles, checking the alpha channel actually survives to disk.
- **Blender version matrix.** Already in CI for 5.1 and 5.2; the robustness
  additions should ride along, since `bpy` API drift is the most likely source
  of a regression nobody wrote.

### Tier 4 — exploratory, performed by an LLM agent

Where the value is in *inventing* the input rather than in re-running a fixed
one. An agent with a Blender it can drive, asked to:

- read a module and try to falsify each documented claim in its docstrings
  (this document's §5 was produced exactly that way);
- drive the UI as a confused user would: empty fields, a selection naming an
  alias that was just deleted, an operator run in Edit Mode with the mesh in a
  state the panel did not anticipate, undo in the middle of a multi-step
  operation;
- cross-check Gala against PyMOL itself on the selection language, where a real
  PyMOL is available — the language is a compatibility claim, and only a
  differential test can check it;
- take a real deposited structure with awkward features (altlocs, insertion
  codes, negative residue numbers, multiple models, zero-occupancy atoms,
  UNK residues, D-amino acids) and run the whole public API over it;
- review a diff for newly introduced boundaries that none of the tiers cover.

Findings from Tier 4 belong in Tier 1 as fixed cases the moment they are
understood. The agent is a generator, not a gate.

---

## 4. Coverage map

Where each requested category is handled, and by which tier.

| Category | Tier 1 (permanent) | Tier 2/3/4 |
|---|---|---|
| Missing / invalid inputs | Empty and whitespace selection strings; `None`, floats, lists and objects passed as selections; missing CSV columns; missing files; `write_session` to a non-existent directory | T4: operators with no active object |
| Boundary values | `index 0` and `index -5` against 1-based indexing; `within 0`; masks of length n±1; single-node grid; arc `resolution` below 2; a one-atom structure's bounding sphere | T2: property tests at range ends |
| Empty / degenerate | Zero-atom structure across every macro and every expansion level; empty point list to `sample`; empty session, zero-atom molecule and member-less selection round trips; empty CSV | T3: zero-atom molecule through the whole scene pipeline |
| Very large inputs | Parse cache bounded at 512 entries; 10 000-character selection values | T3: 10⁶-atom structures, peak RSS |
| Malformed inputs | 23 malformed selection strings; non-pickles, truncated pickles, corrupt gzip; grids that stop early or disagree with their own header; non-numeric CSV cells | T2: structured mutation fuzzing |
| Unexpected types / shapes | Float arrays and 2-D masks as selections; non-`(n, 3)` point lists; non-string colours; corrupt name lists and views in a session | T2: type-swap mutation |
| Unicode / pathological strings | Non-ASCII chain ids, astral-plane characters, embedded NUL, RTL override, combining marks, 10⁴-character values; Unicode object names through a session round trip; `safe_name` over the same set | T4: names Blender itself rejects or truncates |
| Corrupted files | Truncated, garbage, wrong-suffix and half-gzipped sessions and grids | T2: corpus-driven fuzz |
| Numerical edge cases | `inf`/`nan`/1e308 cutoffs and comparisons; `nan` coordinates; 1e30 coordinates; non-finite grid values preserved; colour conversion outside `[0, 1]`; degenerate arcs and dashes | T3: solver divergence; T2: interpolation properties |
| Repeated / adversarial operations | Parse-cache growth under 2 000 distinct strings | T3: repeated setup/load/alias cycles, datablock leak checks |
| State-machine violations | Operators run after the molecule was deleted, with a template that cannot be formatted, or against a selection that is not there; a failed call leaving half a light rig or an unlinked compositor; repeated setup | T4: Edit vs Object Mode in the GUI, undo after an attribute is destroyed, register/unregister across two installed copies |
| Resource exhaustion | Header-declared grid size not trusted; parse cache bounded | T3: dash-count blow-up, quadratic contact detection, RSS ceilings |
| Nondeterminism | Description output is insertion-ordered and verified before it is returned | T2: repeated-run equality; T3: same scene twice |
| Feature interaction | Describe→select round trips on a real structure; session write→read→write | T3: colour + style + session + measurement in one scene |

---

## 5. What was found, and what was done about it

Every entry below was reproduced by running it — none is hypothetical — and
every one is now fixed and guarded by a test in `tests/test_robustness.py` or
`tests/test_robustness_blender.py`. They are grouped by what they cost the
user, which is not the same as where they lived.

### Data loss and silently wrong results

1. **Storing a selection destroyed the molecule's own data.** An alias named
   `res_id`, `b_factor`, `Color` or `atomic_number` reached `write_boolean`,
   which removes a same-named attribute "of the wrong type" — and on a molecule
   that attribute is Molecular Nodes' per-atom data. `res_id` went from
   `INT [1, 1, 1]` to `BOOLEAN [True, True, True]`, reporting `FINISHED`.
   *Now:* the collision is refused, and Object Mode and Edit Mode agree.

2. **A blank element column invented chemistry.** With no element symbols the
   fallback kept the first *two* letters of the atom name, so `CA` read as
   calcium: `metal_coordination` went from 2 contacts to **15** on `site.pdb`.
   *Now:* `_element_of` follows the PDB convention — the single leading letter
   when it is organic, two only when the two-letter form is a real symbol and
   the one-letter form is not — with a residue-level correction so `NA` in an
   ion residue still reads as sodium. Metal, hydrophobic, polar and hydrogen
   counts are identical with and without the column. One residual, recorded in
   the test: a ligand's `CL1` reads as carbon, because stripped of its column
   it is indistinguishable from a delta carbon.

3. **`publication_setup` from the panel ignored the molecule.** The operator
   resolved the active structure only when it declared `requires_structure`,
   which this one does not, so the origin, material and framing settings in the
   same panel did nothing. *Now:* the structure is resolved whenever there is
   one, and only *required* when the operator says so.

4. **A non-default `scale` put the measurements where the atoms are not.** At
   `scale=0.005` the atoms still spanned 0.09 BU while the 9 Å measurement was
   drawn at 0.045. *Now:* the effective scale is taken from the molecule that
   was actually built, so everything drawn beside it agrees. The parameter
   cannot be honoured for a molecule — Molecular Nodes 5.2's `Molecule.load`
   has no `world_scale` argument and its style node groups bake `0.01` in, so
   rescaling the mesh would give correct atom positions with a cartoon at
   double thickness — so a request that cannot be met is now *reported* rather
   than half-applied.

5. **A negative cutoff widened the search.** `cKDTree` takes the absolute value
   of a negative radius, so `polar_max=-5.0` found 10 contacts where 3.5 Å
   finds 3. *Now:* a cutoff that is not positive finds nothing, both paths
   agree, and every criterion goes through a bounds test that a `nan` fails
   from either side — which also fixes `hbond_angle_min=nan` silently
   *loosening* a detector.

6. **Rings with no plane were reported as stacked.** Six coincident atoms gave
   an arbitrary SVD normal and a confident `pi_stacking … (3.80 A, 0 deg)`.
   *Now:* a ring whose second singular value is at or below 0.05 Å has no
   plane and is dropped, as are rings of fewer than three atoms and rings with
   non-finite coordinates. Separately, an unmerged altloc no longer inflates a
   six-membered ring into a twelve-atom one.

7. **`setup_render` silently discarded every compositing pass** by resetting
   the format to PNG after `set_exr_output`. *Now:* a format that already has
   an alpha channel keeps it, and only its colour mode is corrected.

8. **`render()` returned a path it did not write** — `shot.jpg` while the
   format was PNG wrote `shot.png`. *Now:* it returns
   `scene.render.frame_path(...)`, the name Blender resolved, and the first
   frame's when rendering an animation.

9. **A spreadsheet CSV with a byte-order mark was unreadable.** *Now:* read as
   `utf-8-sig`, and a cell that is not a number reports the file, the line
   number, the column and the offending text.

10. **`hex_to_rgb` accepted near-misses** — `"+12345"`, `"-fffff"` and
    fullwidth `"１２３４５６"` were read as colours, two of them with negative
    channels. *Now:* only the sixteen ASCII hex digits are a colour.

11. **A file that was not an image was accepted as an HDRI**, giving a world
    lit by a zero-sized image. *Now:* refused once the load can be seen to have
    failed.

12. **`_net_charge` summed whatever sat in the second-to-last field** of any
    line starting `ATOM`. *Now:* the full PQR atom layout is required and the
    last five fields must all read as finite numbers.

13. **Group hierarchy was lost on load and group membership on save.** *Now:*
    collections are created and then linked into their parents, with a cycle or
    a missing parent resolving to the scene; and `scene_to_session` records each
    molecule's collection, excluding Molecular Nodes' own and Gala's.

14. **A state index that did not exist was only checked for multi-state
    objects.** *Now:* checked for every object of the session.

15. **A selection whose name collided with a mesh attribute** aborted the load
    from inside `databpy`. *Now:* one that would collide with an attribute of
    another domain or type is stored as `pymol_<name>` and the rename reported.

16. **`safe_name` mapped distinct names onto one** — `!!!`, `@@@ ###`, `éèê`
    and `中文选择` all became `selection`. *Now:* a name with no word
    characters keeps a short digest, so distinct names stay distinct and a
    CJK-only name is usable.

17. **An alias named after a language keyword was unreachable** while the panel
    said otherwise. *Now:* the panel quotes the form that actually reaches the
    selection — `%protein`, not `protein`.

### Bare exceptions where a reported error was intended

18. **A `nan` coordinate broke every spatial selection** with scipy's
    `data must be finite` — reachable from any multi-state PyMOL session, since
    absent atoms are stored as `nan`. *Now:* the KD-tree is built from the
    atoms that have positions and the rest simply never match; non-spatial
    selections are unaffected, and the fast path is unchanged when every atom
    is placed. The same reading is now taken by `bounding_sphere` and by the
    camera's point fit, so a session state with missing atoms frames and lights
    the atoms it does contain. Only a structure with *nothing* placed is
    refused.

19. **Deleting the molecule broke every operator.** Molecular Nodes'
    `Molecule.object` raises `LinkedObjectError`, not `AttributeError`, so the
    liveness filter never applied and the call sat outside `execute`'s `try`.
    *Now:* reported, not a traceback.

20. **One degenerate measurement aborted a whole session load** — PyMOL writes
    `dist d, x, x` without complaint. *Now:* collected into `skipped` with
    every other per-item failure.

21. **The selection parser exhausted the Python stack** at ~200 nested
    parentheses or ~2 000 `or` clauses. *Now:* nesting is bounded and reported
    as a syntax error, and `and`/`or` are n-ary nodes evaluated with a loop, so
    a long flat chain — which is legal — simply works.

22. **A corrupt session escaped as a bare exception** for a malformed name
    list, a non-numeric view, or a corrupt gzip header decompressed outside the
    guard. *Now:* all `PymolSessionError`.

23. **`bounding_sphere()` on a structure with no atoms** raised numpy's
    `zero-size array to reduction operation maximum`. *Now:* `StructureError`.

24. **`{0}` in a label template escaped as a traceback** because `IndexError`
    was not in the caught tuple, while every other malformed template was
    reported. *Now:* caught — as is the `IndexError` `gala.color` raised once
    the mesh vertex count drifted from the atom count.

25. **Deleting a selection that was not there reported success.** *Now:*
    `{'CANCELLED'}`.

26. **Non-mesh objects, non-string presets and unknown material names** gave
    `AttributeError`/`KeyError` from `bpy`. *Now:* `StructureError` and
    `ValueError`s that name the alternatives.

27. **`read_dx` accepted a grid that was not three-dimensional**, failing much
    later inside `sample`. *Now:* refused at the point of reading.

28. **The documented negative probe radius did not work** — `probe=-2.0` gave
    biotite's `Cell size must be greater than 0`. *Now:* the reach is clamped
    at zero so the sample sphere collapses to the atom centre, as documented,
    and the cell size is floored separately; `inf` and `nan` are refused.

29. **`potential_at_atoms(points=0)`** returned all-`nan` silently and then
    made `summary()` raise. *Now:* `points` is validated and `summary()` says
    "no value at any of N atoms".

30. **`selected_atom_indices` read a different object than the operator acted
    on**, with no atom-count check, so a picking on an unrelated mesh measured
    the molecule. *Now:* checked and reported.

31. **`set_origin` run from Edit Mode teleported the molecule**, writing a
    stale `obj.data.vertices` while still shifting `matrix_world`. *Now:* it
    goes through `viewport.object_mode()`, as `style_alias` already did.

32. **Panels drew `context.scene.gala` unguarded** while `poll` checked only
    for a scene. *Now:* `poll` checks the property group.

### A failed call left the scene half-rebuilt

33. **`depth_cue` with `near > far`** removed the compositor nodes before
    validating, leaving an unlinked output. **`three_point_lighting` with a
    malformed `LightSpec`** cleared the old rig first and validated each spec
    only as `bpy` applied it. **`draw_interactions`** aborted on a degenerate
    contact after creating objects for the earlier ones. *Now:* all three
    validate before they destroy, and `draw_interactions` skips the degenerate
    ones with a warning naming them.

34. **`orbit` displaced the camera by the target centre**, reading the pivot's
    parent inverse before the depsgraph had updated — the opposite of the
    framing it documents — and `orbit(0)` collapsed the scene range to `0..0`.
    *Now:* the update happens first and a non-positive frame count is refused.

### Also fixed, without a test of their own

35. **Unbounded work.** `apbs._run` now takes a timeout (an hour by default,
    plumbed through `run_apbs(timeout=)`) instead of wedging Blender's main
    thread on a hung solver, and streams child output to the log file rather
    than buffering it all in memory. `find_executable` requires a *file*, so a
    directory named `apbs` no longer becomes a `PermissionError`.

36. Smaller: `frame_target` requires a positive margin; `select` reports an
    out-of-range atom index rather than raising `IndexError`; a value list with
    no values (`resi +`) is a syntax error; `plip._convert` treats a
    non-numeric field as absent instead of aborting, and can no longer resolve
    both sides of an interaction to the same atom; a duplicate object name in a
    session no longer orphans the first molecule; an object name over 250
    characters is truncated rather than dropped; and `"could not be written for
    import"` no longer covers both a filesystem failure and Molecular Nodes
    being unregistered.

### Still open

- **Quadratic detection.** `atom_contacts` on 3 000 coincident atoms still
  produces 4.5 M `Interaction` objects, and `salt_bridges` has no spatial
  index. Both need a bounded-work policy rather than a guard, which is a design
  decision, not a bug fix. *Tier 3.*
- **`dash_segments` has no cap** on the number of dashes, so a 10⁴ Å line with
  a 10⁻³ dash builds five million segments. Non-finite endpoints are refused,
  but by `round()` rather than deliberately.
- **`save_session` has no `groups` toggle** to match `load_session(groups=…)`.
- **`plip._convert`'s fixes are unverified end to end** — PLIP is not
  installable in this interpreter, so they were exercised against hand-built
  stand-in records only.

## 6. Running it

```sh
make test                       # everything, inside Blender — includes tier 1
make test-fast                  # only what needs no Blender objects
blender --background --python tests/run_tests.py -- -q \
    tests/test_robustness.py tests/test_robustness_blender.py
```

Everything in §5 is fixed and guarded, so the suite is plainly green: there are
no `xfail` markers left in either module. When the next gap is found, record it
as a strict xfail first — the `XPASS(strict)` failure is what tells you the fix
worked.

---

## 7. Structures that came from somewhere real

The fixtures the rest of the suite uses are built to be *convenient*: one
chain, consecutive residue numbers, one conformation, an element column on
every atom. Nothing deposited in the PDB looks like that for long, and the
complications are not exotic — they are the normal state of a
high-resolution structure.

### What "diverse" means here

| Axis | What varies | Where it is covered |
|---|---|---|
| **Format** | `.pdb` against `.cif` for the *same* structure. mmCIF carries multi-character chain identifiers, `label_*` alongside `auth_*`, and no convention for guessing a missing element | `awkward.pdb` / `awkward.cif` |
| **Numbering** | an expression tag below 1, a gap where a loop was not modelled, insertion codes (`100`, `100A`, `100B`) as antibodies number them | `awkward.pdb` |
| **Conformation** | alternate locations at partial occupancy — two atoms of the same name in one residue | `awkward.pdb`, and isolated arrays |
| **Composition** | protein, DNA, RNA, selenomethionine, `UNK`, glycans, sulfate, a metal, water | `awkward.pdb`, `nucleic.pdb` |
| **Ambiguity** | a calcium named `CA` beside C-alphas named `CA`; a chain `a` beside a chain `A`; an atom duplicated at identical coordinates | `awkward.pdb` |
| **Completeness** | a zero-occupancy atom, a missing element column, formal charges | `awkward.pdb` |
| **Models** | three of them, as an NMR ensemble is written | `ensemble.pdb` |
| **Scale** | past the PDB format's five-digit atom and four-digit residue fields | isolated arrays; the full-size case is Tier 3 |

Every fixture is generated by `tests/data/make_fixtures.py` from exact
coordinates, so CI's existing "fixtures are reproducible" check covers them and
the permanent suite still needs no network.

### The check that earns its place

`describe_selection` renders a mask back into the selection language, and its
output must re-select the same atoms. It is run over every macro and over
*every single atom* of every awkward fixture. That one assertion is what
catches a structure whose chemistry cannot be named unambiguously — two atoms
of a name in one residue, a blank chain, an insertion code — because the
function is required to notice and fall back to the positional form rather
than emit a description that means something else. It self-verifies, so the
test is really testing the fallback.

### Tier 3: the real thing

`scripts/survey_structures.py` fetches real entries from the RCSB and runs the
API over each. It needs the network, so it never runs in CI by default; it
exits non-zero when a structure fails a check, so it can run on a schedule.
The synthetic fixtures encode what the survey has already found; the survey is
what finds the next thing.

### Found this way

Everything below was found by pointing the API at an awkward structure, and is
now fixed and guarded unless marked otherwise.

1. **`write_pdb` silently renumbered structures past the PDB format's limits.**
   126,000 atoms wrote 63,000 residues as 9,999 distinct numbers and restarted
   atom serials at 1, with only a Python `UserWarning` — which does not reach
   the Blender console. PDB2PQR then read roughly twelve atoms per residue and
   `run_apbs` returned a potential for a chemically impossible molecule.
   *Now:* hybrid-36 numbering when the plain fields overflow, which is lossless
   and round trips; a `GalaError` naming the field and the limit beyond that;
   and `run_apbs` refuses up front, since PDB2PQR reads plain integers.

2. **An unmerged alternate conformation silently deleted an aromatic ring.**
   The ring-size gate added earlier to stop two conformers fusing into one
   twelve-atom ring over-corrected: it dropped the residue and marked it
   handled, so geometric perception could not recover it either, and a real
   π-stack was reported as nothing. The same input *fused* a carboxylate,
   reporting a salt bridge at 3.40 Å where the conformer sits at 3.54 Å — a
   distance no conformer occupies. Altlocs also produced two interaction
   records with byte-identical labels. *Now:* one `primary_conformer` mask —
   highest occupancy per (residue, atom name), ties broken deterministically —
   is applied in perception and in `detect._resolve`, so fusion is structurally
   impossible rather than gated. This is what PLIP does. The consequence, and
   it is a real one: an interaction made only by the minor conformer is not
   reported.

3. **`elem C` matched nothing when the element column was blank**, while the
   macro `carbon` was right — an earlier fix covered the macro path and left
   the property path reading the raw annotation.

4. **A residue in no name table was matched by no macro at all** — not
   `polymer`, not `hetero`, not `ligand` — so `UNK` and modified nucleotides
   were reachable only by name or index. Molecular Nodes stores flags that
   Gala prefers, so the shipped add-on was right and the headless science
   layer, which this whole suite runs on, was blind. *Now:* an unknown residue
   is read by its backbone composition.

5. **`dna` and `rna` were exact synonyms of `nucleic`**, so `color orange, rna`
   coloured the DNA.

6. **`AtomStructure(array=stack)` did not reduce the stack** despite its
   docstring: `n_atoms` reported the model count, and `.coord` and
   `.context.coord` read different models. `from_any(structure, frame=N)` also
   ignored the frame.

7. **`write_pdb` raised biotite's `BadStructureError`** — not a `GalaError`, so
   it escaped as a traceback — for a multi-character chain identifier or a
   frame far from the origin. Both arrive from ordinary mmCIF.

### Found by the survey of real entries

The synthetic fixtures encode what is already known; the survey is what finds
the next thing. Nineteen entries, read through both the bare-biotite path and
Molecular Nodes' readers, in both formats where both exist.

8. **Chain identifiers differing only in case were silently merged.** Every
   value in the language is upper-cased before matching, so `chain A` matched
   5568 atoms of 6J5K where the file has 3869, and 72 of its 120 author chains
   were affected; on 4V6X, `byres (chain AZ)` dragged in 6673 atoms of chain
   `Az`. mmCIF `auth_asym_id` is case-sensitive, and using both cases is
   exactly how an assembly gets past the 62 single-character identifiers. *Now:*
   identifier-like keywords match case-sensitively first and fall back to
   case-insensitive only when nothing matched — so `chain A` on a file holding
   both selects only `A`, while `chain d` on a file holding only `D` still
   finds it.

9. **`alt` was a keyword that could never match.** No reader passes
   `altloc="all"`, so the `altloc_id` annotation is never attached, and a
   missing annotation resolved to an all-empty string array: `alt A` returned
   nothing on every structure, including the ones that really do have two
   conformations. `charge` did the same on anything read from mmCIF, and `ss`
   likewise. *Now:* asking for data the structure does not carry is reported,
   which is a different fact from "no atom matched".

10. **An empty query perceived the whole structure.**
    `find_interactions(s, "none", "none", kinds="all")` on a 238,000-atom
    assembly returned an empty list in **46.5 s** — 10,668 rings paired 57
    million ways for an answer that was empty before it started, on Blender's
    main thread. *Now:* it returns early, and the rings are filtered against
    the selections before they are paired.

### Found, and deliberately not changed

- **`metal_coordination` treats its two selections as a union** where every
  other detector requires both sides. `find_interactions` now returns nothing
  when either side is empty, which is what "interactions *between* two
  selections" means, so the union is visible as an inconsistency rather than
  hidden by it. Calling `metal_coordination` directly is unchanged.
- **`describe_selection` is quadratic in residue count** — 8.7 s over 63,000
  residues in one chain, against 0.03 s for the selection itself. *Tier 3*, and
  the right assertion is a wall-clock ceiling: under 1 s for `name CA` over
  60,000 residues.
- **biotite 1.7.1 raises `ValueError: assignment destination is read-only`**
  whenever `altloc="all"` meets an atom whose element must be guessed. Upstream,
  reproducible in four lines, and it means the read-every-conformation path is
  unusable on any file with a missing element column.
- **The two formats disagree about a missing element.** The PDB reader guesses
  it from the atom name, as the format's convention allows; the mmCIF reader
  leaves it unset, because mmCIF has no such convention. Pinned by a test so a
  change to either reading is noticed rather than found in a figure.
- **Modified nucleotides are classified as ligands on the headless path, and
  the macros mean different things depending on the reader.** On 1EHZ (tRNA),
  bare biotite gives `nucleic` = 1329 atoms where biotite's own
  `filter_nucleotides` says 1652; the 323 missing atoms — `1MA`, `2MG`, `5MC`,
  `PSU`, `YYG` and friends — all come back as `ligand`, so
  `find_interactions(mol, "ligand", "nucleic")` reports the RNA's own bases as
  the bound ligand. Through Molecular Nodes' readers the same file gives
  `nucleic` = 1652 and `ligand` = 0, because Gala prefers the `is_nucleic` flag
  when it is there. The composition fallback added for `UNK` does not reach
  these: they are written as HETATM, and the `hetero` gate is what keeps a
  nucleotide-*like ligand* (ATP, AMP) out of `nucleic`. Telling a modified base
  in a chain from a nucleotide ligand needs connectivity or chain context, so
  this is a design decision rather than a patch, and the reader-dependence is
  the part that should be settled first.
- Blank chain, altloc and insertion-code values cannot be selected positively
  (`chain ""` matches nothing), though the negated forms work; and `charge`,
  `b` and `q` take no value list. Both match PyMOL.

---

## 8. Molecules coming, going, changing and multiplying

Every Blender test in the suite used to load exactly **one** molecule, once,
and leave it alone until the test ended. That is not a scene anyone actually
builds. A user imports a structure, imports a second to compare against,
duplicates one with Shift+D, tabs into Edit Mode and prunes a few atoms, scales
an object to fit the layout, deletes the first structure, reloads it — and
presses buttons in the sidebar at every point in between.

Three axes, one module each.

### The lifecycle: loading, reloading, deleting

Molecular Nodes' session outlives the objects it tracks, so "the molecule is
gone" is a state Gala meets routinely.

1. **A dead session entry poisoned every live one.** Resolving *any* object
   walks the session, and reading `entity.object` on an entry whose object was
   deleted raises `LinkedObjectError` — not an `AttributeError`, so the
   `getattr` default never applied and the walk died on the first dead entry,
   whichever live object had been asked for. The identical hazard §5.19 fixed
   in the operator layer, on the library path that `select`, `distance`,
   `label`, `find_interactions`, `color_by_*` and `create_alias` all funnel
   through — 40+ call sites. Order-dependent: deleting the earlier-loaded
   molecule poisoned the later, deleting the later left the earlier fine. Both
   orders are now asserted.
2. **A stale structure raised `ReferenceError`** from `.name`,
   `.world_positions()`, `store_alias` and the rest — including from `repr`,
   so an error message interpolating `self.name` turned one failure into two.
   Now a `StructureError` naming what went, with a `repr` fallback; the pure
   chemistry (`n_atoms`, `coord`, `atom_label`) keeps answering, because the
   array never needed the object.
3. **Whether a stale structure could be selected from depended on cache
   history** — two scripts differing only in an earlier, unrelated call
   disagreed. Now they agree, and the test compares the two paths against each
   other rather than letting each have its own answer.
4. **`select` could not take a Blender object** while `distance` and `label`
   could.

### Modification: the vertex-to-atom correspondence

Gala's central assumption is that vertex *i* is atom *i*.

5. **A restored vertex count was taken as proof.** Delete five vertices, add
   five back: the count matches, the guard does not fire, every atom reads a
   different vertex. A 1.41 Å bond reported as **100 Å**; `color_by_selection`
   painting seven vertices for a two-atom selection; success reported both
   times. The guard now checks *identity* — the mesh's `atom_id` and `res_id`
   against the array's — which catches deletion, addition and reordering.
6. **The object's transform scaled measurements documented in ångström.**
   `scale=(2,2,2)` turned a 1.414 Å bond into 2.828 Å, `(3,1,1)` turned a
   115.3° angle into 61.5°, and a mirrored object flipped a dihedral's sign.
   Meanwhile `find_interactions` measured on the array, so the two disagreed
   about the same molecule. Measurement now happens in the molecule's own
   frame — the transform is divided out rather than the object refused, since a
   non-uniform scale is a legitimate layout choice and the bond length is still
   well defined.
7. **A viewport selection of 20 vertices read back as `'none'`**, and
   `create_alias` answered "select some atoms in Edit Mode first" — advice the
   user had just taken.
8. **The colour path validated against the mesh and never the atoms**, so an
   atom-indexed array was written onto vertex indices whenever the two lengths
   happened to agree; `read_colors` returned one row per vertex against a
   docstring promising one per atom; and a genuine mismatch escaped as raw
   numpy.
9. **`frame=` reached the chemistry and not the geometry** — the base mesh is
   always model 0, so everything *drawn* landed on the wrong model. The same
   split fixed between `.coord` and `.context.coord` a round earlier,
   reappearing one level out.
10. **Moving the origin displaced everything drawn** once anything else made
    `local_positions` fall back to the array — measured at 4.4 Å, with labels
    and interaction lines drawn there.

### Several at once

11. **`publication_setup` pulled its target out of register with every other
    molecule.** `move_to_world_origin` throws away the compensation that keeps
    the geometry where it was, and moves only the target: a protein and its
    separately imported ligand went from 2.28 Å apart to 3.52 Å, the ligand
    left the pocket, `warnings` was empty and the operator reported success.
    Now the rest of the figure travels by the same delta — including Gala's own
    labels, measurements and interaction lines, whose world-space anchors are
    shifted with them; the camera and lights deliberately stay, since bringing
    the subject to a rig built about the origin is what the flag is for.
12. **A Shift+D copy retargeted every operator to the original.** Resolving the
    untracked mesh raised, the operator layer swallowed it, and the
    "only molecule in the session" fallback applied — so colours, aliases,
    labels and measurements landed on the object the user was not looking at.
    An active mesh that fails to resolve is now refused; nothing active, or a
    non-mesh active, still resolves the lone molecule.
13. **Colouring one copy recoloured the other**, because a duplicate shares its
    node group and muting `Set Color` changes what both render. Two separately
    *loaded* copies get separate trees and were always fine — it is duplication
    that shares. The target's tree is now made its own first.
14. **`save_session(molecules=[a])` adopted the other molecule's annotations**
    — a label 100 Å away recorded on atom 71 of A, and measurements ignoring
    the filter entirely.
15. **A label on one of two superposed copies was attributed to the other.**
    Two separately loaded superposed molecules leave *no* signal in the scene:
    identical coordinates, identical transforms, an anchor exactly zero from
    both. Labels now record which molecule they were drawn from, and an anchor
    that genuinely cannot be attributed is skipped rather than guessed.
16. **Setting up a scene with no resolvable subject said nothing** — materials
    and the origin were skipped with an empty `warnings`, the only field the
    operator surfaces.

### Recorded, not fixed

- **There is no way to ask for interactions between two objects.** A protein
  and its ligand imported separately is the most common real case, and
  `find_interactions` takes a single target; naming an atom from the other
  object gives a correct, well-worded `EmptySelectionError` that cannot be
  satisfied. Concatenating the two arrays works and reproduces the one-file
  answer exactly, but carries file coordinates rather than each object's
  transform, so it is a workaround rather than the feature. This is a
  capability gap, not a defect.
- **Named selections restored from a PyMOL session are not registered as
  aliases**, so the Stored Selections panel is empty after a session load even
  though the attributes are there and `%name` resolves.
- **`save_session` applies the first molecule's world scale to every
  measurement, label and the camera.** Latent — nothing in Gala writes a
  non-default scale, and Molecular Nodes has no scale argument.
- **The origin offset is recovered by inference, not bookkeeping.** Nothing
  records it, so the fallback matches `atom_id` and takes the median offset,
  accepting it only if a majority agree. Recording the offset where it is
  applied would be sturdier.
