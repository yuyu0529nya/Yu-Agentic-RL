import json

with open("e:/yuyu/third_party/tau2-bench/data/simulations/airline_baseline_50_anthropic_glm_5_1_50tasks_1trials/results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

targets = {"44", "2", "37", "23"}
output = []

for sim in data["simulations"]:
    tid = str(sim["task_id"])
    if tid not in targets:
        continue

    reward = sim["reward_info"]["reward"]
    msgs = sim["messages"]

    output.append(f"\n{'='*80}")
    output.append(f"TASK {tid} | reward={reward} | termination={sim.get('termination_reason')}")
    output.append(f"{'='*80}")

    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        content = m.get("content", "")
        tool_calls = m.get("tool_calls")

        out_parts = []

        # Text content
        if isinstance(content, str) and content.strip():
            out_parts.append(content[:1200])
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        out_parts.append(c["text"][:1200])
                    elif c.get("type") == "tool_use":
                        inp = json.dumps(c.get("input", {}), ensure_ascii=False)[:500]
                        out_parts.append(f"[TOOL_USE: {c.get('name','?')}({inp})]")
                    elif c.get("type") == "tool_result":
                        rc = c.get("content", "")
                        if isinstance(rc, list):
                            rtext = " | ".join(
                                x.get("text","")[:300] if isinstance(x, dict) else str(x)[:300]
                                for x in rc
                            )
                        else:
                            rtext = str(rc)[:500]
                        out_parts.append(f"[TOOL_RESULT: {rtext}]")

        # Tool calls from top-level field
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tc_name = tc.get("function", {}).get("name", tc.get("name", "?"))
                    tc_args = tc.get("function", {}).get("arguments", tc.get("input", {}))
                    if isinstance(tc_args, str):
                        try:
                            tc_args = json.loads(tc_args)
                        except:
                            pass
                    args_str = json.dumps(tc_args, ensure_ascii=False)[:500] if isinstance(tc_args, dict) else str(tc_args)[:500]
                    out_parts.append(f"[TOOL_CALL: {tc_name}({args_str})]")

        content_str = "\n".join(out_parts)
        output.append(f"\n--- [{i}] {role} ---")
        output.append(content_str)

with open("e:/yuyu/reports/task_44_2_37_23_traces.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(output))

print(f"Done. {len(output)} lines written.")
