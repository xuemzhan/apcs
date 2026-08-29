## Conclusion
Cache translation promises to move a teacher's work into a smaller model. The
audit separates that promise into three testable parts. The mechanism works:
an identity control reproduces the student's own prefill exactly. The map
does not: five mapper families, a calibration ladder, and an
architecture-anchored translator all leave the strong student below its own
prefill, and the weak student at parity. The reason is not the map: oracle
probes bound the student-readable advantage of the teacher's cache at zero,
and a near-native-perplexity cache still fails the task. The teacher's
capability is in its weights. We release the audit framework so that the
next cache-translation claim can be checked in one run.
plainnat
references