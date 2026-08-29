## Why Translation Fails: An Architectural Analysis
sec:why
The audit's verdicts fit the architecture. A cache entry is a linear
projection of a residual-stream state . Translation must recover
 from , move it across the residual-space gap ( to ), and
re-project. Two facts bound the recovery. First, the pullback
 returns only the row-space component of ; the component
orthogonal to 's rows, which can dominate a -dimensional state
observed through a -dimensional head, is invisible to the teacher's own
cache and therefore unavailable to any translator, however nonlinear. The
student's corresponding value depends on that invisible
component, so is not a function of : the mapping task is
intrinsically underdetermined, and a learned translator can at best track the
conditional mean. Second, the teacher's advantage over the student lives in
weights, not in caches: FFN layers carry the model's factual
associations~ffnmemory2021, and the capability gap between Qwen3-4B
and 1.7B is a gap in these parameters, which a cache does not transport. The
oracle probes make the consequence concrete: even with the student's own
cache holding of the mass, the teacher's contribution degrades
the answer. MoT's own design corroborates the analysis: its reported quality
includes a target-side replay with a correction loss, which is exactly the
upper-layer correction that strict zero re-prefill denies. Under a cold
start, the translator's output is the whole story, and the whole story does
not contain the teacher's capability.