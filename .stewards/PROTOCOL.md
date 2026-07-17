# Steward Review Protocol

This file is on-demand governance guidance for explicit review, audit, and
steward-network maintenance. It is not part of ordinary implementation context.

## Review flow

1. Read the root map and only scoped maps for affected paths.
2. Identify applicable invariants and exact focused checks before exploring broadly.
3. Run or trust green machine-backed checks; spend judgment on manual, unenforced,
   and cross-domain concerns.
4. Record findings with evidence, user impact, required fix, proof, collateral,
   confidence, and verification status.
5. Preserve minority reports when stewards disagree; the implementing agent owns
   synthesis.
6. Human CODEOWNERS approve. Stewards advise.

## Finding format

```text
Steward:
Area:
Severity: P0/P1/P2/P3
Invariant:
Evidence: <source-file:line> [-> <doc-file:line>]
User Impact:
Required Fix:
Required Proof:
Collateral:
Confidence:
Verification Status: machine-verified / manual-confirmation-needed / not-machine-verifiable
```

Independent matching findings increase confidence and require a shared source
verification. Severity follows demonstrated user or contract impact, not vote count.
Accepted P0s require a bounded repository-wide sibling-pattern sweep before closure.

Cross-surface changes include Steward Notes naming consulted maps, accepted and
deferred findings, proof run, collateral updated, unresolved dissent, and any
explicit no-impact rationale.

When an escaped bug shows that guidance was missing, update the smallest useful
invariant, check, routing rule, regression test, or explicit reason not to encode
policy. When guidance repeatedly overreaches or produces noise, narrow or prune it
instead of accumulating more instructions.
