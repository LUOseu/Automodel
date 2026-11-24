import json
from pathlib import Path

folder = Path("/dev/shm/prc-pilot/data")
rows = [json.loads(x) for x in (folder / "ultrachat_2048.jsonl").read_text().splitlines()]
assert len(rows) == 2048
with (folder / "ultrachat_2048_first_turn.jsonl").open("w") as stream:
    for row in rows:
        messages = row["messages"]
        end = next(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        assert any(m["role"] == "user" for m in messages[:end])
        stream.write(json.dumps({"messages": messages[:end + 1]}) + "\n")
