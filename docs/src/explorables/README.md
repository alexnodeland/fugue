# Explorables

Most explanations of probabilistic programming ask you to read first and understand later.
These pages invert that. Every figure here is a small machine: drag the numbers, click the
data, step the algorithms, and watch the mathematics respond. You will have *felt* why a
sampler stalls or a particle filter collapses before you meet the equation that says so.
The genre is borrowed with gratitude from Bret Victor's explorable explanations and
3Blue1Brown's visual mathematics.

Two habits worth forming:

- **Touch everything.** If a number looks interesting, try dragging it — dashed underlines
  mark the ones that respond. Canvases are clickable more often than not.
- **Mind the seed.** Wherever randomness appears, a scrubbable seed appears with it. The same
  seed always replays the same run — which is not a gimmick, it is fugue's worldview: a
  recorded run is a trace, and traces can be replayed. You will meet this idea everywhere.

One color language runs through every page:
<span class="fv-c-prior">**prior**</span> ×
<span class="fv-c-data">**likelihood**</span> =
<span class="fv-c-post">**posterior**</span>, with
<span class="fv-c-hot">**coral**</span> for the current sample and
<span class="fv-c-flow">**violet**</span> for momentum and structure.
Once you know it, every canvas and every equation on this site reads at a glance.

## The six machines

1. **[Anatomy of a Probabilistic Program](./anatomy.md)** — a coin, a prior you can bend,
   and data you can click into existence. Bayes' rule as something your hands learn first.
2. **[The Model Is a Score](./monad.md)** — five observations you can drag along an axis,
   a prior, and an exact posterior that follows your hand. Underneath, fugue's `Model` monad
   performed one effect at a time — then handed to a different performer. The page that
   explains fugue itself.
3. **[Random Walks in Posterior Space](./metropolis.md)** — a real regression: drag the data
   points and watch the posterior heatmap deform under the sampler's feet, while accepted
   samples paint fit-lines through your data. Live split-R̂ and ESS referee the whole thing.
4. **[Rolling, Not Guessing: Hamiltonian Monte Carlo](./hmc.md)** — the same regression,
   but the sample gets momentum and rolls. Leapfrog trajectories, divergences, and a
   side-by-side race against the random walk at a matched budget — visible in the data,
   not just the parameters.
5. **[Particles That Tell Stories](./smc.md)** — Sequential Monte Carlo as a population:
   propagate, weight, resample. Watch lineages go extinct and understand degeneracy by
   witnessing it.
6. **[A Field Guide to Distributions](./distributions.md)** — every distribution fugue ships,
   with real parameterizations, a sampler racing its own density, and the natural return
   types that make fugue's models type-safe.

Read them in order for a course, or jump to what you came for — each page stands alone and
links onward. When a page convinces you, the matching tutorial turns the intuition into
working Rust; the code on every explorable compiles against `fugue-ppl 0.2.0` as written.
