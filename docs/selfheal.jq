# Pretty-print an /api/trace/:id response as a Self-Heal Report.
# Usage: curl -sS localhost:8000/api/trace/$ID | jq -rf docs/selfheal.jq
"query:     \(.query)",
"triggered: \(.self_heal_triggered)   failed checks: " +
  ([.trigger_failed_thresholds[] | "\(.name) \(.value) < \(.threshold)"] | join(", ")),
"attempts:  \(.attempts | length), kept #\(.best_attempt_no)",
(.heals_applied[] |
  "heal:      \(.type) -> \"\(.params.chosen_rewrite // (.params | tojson))\"",
  "delta:     max_fused +\(.delta.max_fused)  term_overlap +\(.delta.overlap_top3)  separation +\(.delta.score_separation)")
