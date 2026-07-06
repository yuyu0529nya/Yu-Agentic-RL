# Phase2J Gated Policy Upper Bound

## Goal

Estimate whether separating action decision from tool generation is worth training.

## Reported 64-Probe Metrics

| Policy | Action type acc | Text no-tool | Tool-call rate | Wrapper | Tool-name acc | Arg exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.531 | 1.000 | 0.062 | 0.062 | 0.062 | 0.062 |
| Phase2H | 0.734 | 0.719 | 0.750 | 0.750 | 0.750 | 0.750 |
| Phase2I | 0.703 | 0.812 | 0.594 | 0.594 | 0.531 | 0.531 |
| Phase2I gate + Phase2H tool | 0.672 | 0.812 | 0.531 | 0.531 | 0.531 | 0.531 |
| oracle gate + Phase2H tool | 0.875 | 1.000 | 0.750 | 0.750 | 0.750 | 0.750 |

## Deduplicated Metrics

| Policy | N | Action type acc | Text no-tool | Tool-call rate | Wrapper | Tool-name acc | Arg exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 55 | 0.600 | 1.000 | 0.043 | 0.043 | 0.043 | 0.043 |
| Phase2H | 55 | 0.727 | 0.719 | 0.739 | 0.739 | 0.739 | 0.739 |
| Phase2I | 55 | 0.745 | 0.812 | 0.652 | 0.652 | 0.565 | 0.565 |
| Phase2I gate + Phase2H tool | 55 | 0.709 | 0.812 | 0.565 | 0.565 | 0.565 | 0.565 |
| oracle gate + Phase2H tool | 55 | 0.891 | 1.000 | 0.739 | 0.739 | 0.739 | 0.739 |

## Interpretation

- Phase2H is still the stronger tool generator.
- Phase2I is a better decision signal than Phase2H for text turns, but it suppresses too many true tool calls.
- The oracle gate shows the target: keep Phase2H tool accuracy while making text no-tool perfect.
- Next implementation should train a decision-only gate instead of another one-stage mixed SFT.
