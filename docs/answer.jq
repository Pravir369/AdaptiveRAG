# Pretty-print an /api/chat response: the answer, then the documents it cited.
# Usage: curl -sS localhost:8000/api/chat ... | jq -rf docs/answer.jq \
#          --argjson d "$(curl -s localhost:8000/api/docs)"
.answer,
"",
"sources: " + (
  [.citations[].doc_id] | unique
  | map(. as $id | $d.docs[] | select(.doc_id == $id) | .title | sub("\\.txt$"; ""))
  | join(", ")
)
