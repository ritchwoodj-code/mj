# NBA 2K26 Shooting Guard — Optimal Build Run Plan

This doc is the executable spec for finding the optimal PS5 MyPlayer **Shooting Guard**
using **live, tested cap data** from the simulator. No cap may be guessed or estimated —
every number in the final output must be read from the tool (or a cited source).

## Why this doc exists
The first session had **no browser-automation MCP** and both data sites returned **HTTP 403**
to static fetch, so the live sweep could not run. `.mcp.json` now configures Playwright MCP.
**Start a new session** so the harness boots the browser server, then execute the steps below.

## Pre-flight (do before sweeping)
1. Confirm Playwright MCP tools are present (`browser_navigate`, `browser_snapshot`, `browser_click`, etc.).
2. Navigate to `https://2kcourtvision.com/build-creator`.
   - If it returns 403 / Cloudflare challenge even via the real browser, **STOP and report** —
     the environment network policy likely blocks the host. Do not substitute guessed numbers.
3. Cross-check at least one frame against `https://www.nba2klab.com/myplayer-builder`.
   If it needs an account or blocks the browser, report that instead of guessing.

## Target stat line
| Attribute        | Target |
|------------------|--------|
| Three-Point      | 95 |
| Driving Dunk     | 93 |
| Driving Layup    | 88 |
| Pass Accuracy    | 79 |
| Speed            | 90 |
| Acceleration     | 90 |
| Vertical         | 80 |
| Perimeter Defense| 89 |
| Steal            | 90 |
| Speed With Ball  | 89 |
| Ball Handle      | 89 |

## Priority order (protect in this order if not all fit)
1. Three-Point 95
2. Perimeter Defense 89
3. Speed 90
4. Driving Dunk 93
5. Everything else

## Sweep matrix
- Position: **Shooting Guard**. Platform: PS5. Current 2K26 season.
- Wingspan rule: max = selected height + 6 inches.
- Heights: 6'4", 6'5", 6'6", 6'7", 6'8".
- At each height, 3 weights: **minimum**, **mid**, **near-max for a guard** (use the tool's live min/max).
- At each height, 2 wingspans: **neutral (= height)** and **max (= height + 6")**.
- => 5 heights × 3 weights × 2 wingspans = **30 frames**.

## Record per frame (read live caps)
Three-Point, Driving Dunk, Speed, Perimeter Defense, Ball Handle, Speed With Ball,
Driving Layup, Vertical. (Also note Acceleration, Steal, Pass Accuracy, Strength where shown.)

## Selection
Pick the single frame satisfying the MOST of the target line, honoring the priority order.

## Required output (all numbers tool-sourced)
1. Table of all 30 frames with the tracked caps.
2. Recommended frame: exact height, weight, wingspan.
3. Full buildable SG attribute sheet — every attribute incl. fillers:
   Close Shot, Standing Dunk, Post Control, Mid-Range, Free Throw,
   Interior Defense, Block, Offensive Rebound, Defensive Rebound, Strength, Stamina.
4. Cap-breaker allocation: which attributes, how many breakers each (max 5/attr), + VC cost if shown.
5. Plain list of target numbers NOT hittable on the recommended frame, with closest achievable value.
6. Build name + badges unlocked at Gold or higher.

## Hard rule
If any value cannot be verified from the simulator or a cited source, say so and stop —
do not fill it in.
