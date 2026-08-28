#!/usr/bin/env python3
"""PLUGS engine: verifies this week's LATEST ecosystem versions actually work together.

CI installs the unpinned stack first (see weekly-benchmark.yml); this script then runs real probes:
  ollama_serve         install/start Ollama, pull tiny model, one real generation (CPU-friendly)
  ollabridge_gateway   start OllaBridge, list models + chat.completions through :11435/v1 (openai SDK)
  agent_via_gateway    LangGraph / LangChain / CrewAI / DeepAgents make one real call via the gateway
  tracing_decorator    Langfuse @observe wraps a gateway call (SDK-level check, no server needed)
  requirements_resolve pip dry-run resolve of an app's requirements.txt against this environment
  import_only          import + version report

Writes {component: {ok, component_version, checked_at, note}} for merge into the registry.
Exit code = number of failed non-baseline components (so CI turns red honestly).
"""
from __future__ import annotations
import argparse, datetime, importlib, importlib.metadata as md, json, os, subprocess, time, traceback
from pathlib import Path
from urllib import request

import yaml

ROOT = Path(__file__).resolve().parents[1]
GATEWAY = os.environ.get("OLLABRIDGE_URL", "http://127.0.0.1:11435/v1")
TODAY = datetime.date.today().isoformat()


def ver(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not-installed"


def http_json(url: str, payload=None, headers=None, timeout=120):
    data = json.dumps(payload).encode() if payload else None
    req = request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    with request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_http(url: str, secs=90) -> bool:
    for _ in range(secs):
        try:
            request.urlopen(url, timeout=2); return True
        except Exception:
            time.sleep(1)
    return False


# ---------------- probes ----------------
def probe_import_only(c):   # noqa: ANN001
    mod = c.get("package", c["id"]).replace("-", "_")
    importlib.import_module(mod)
    return ver(c.get("package", c["id"])), "import ok"


def probe_ollama_serve(c, model: str):
    if subprocess.run("command -v ollama", shell=True, capture_output=True).returncode != 0:
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert wait_http("http://127.0.0.1:11434/api/version"), "ollama not up"
    v = http_json("http://127.0.0.1:11434/api/version")["version"]
    subprocess.run(f"ollama pull {model}", shell=True, check=True)
    r = http_json("http://127.0.0.1:11434/api/generate",
                  {"model": model, "prompt": "Say OK", "stream": False, "options": {"num_predict": 5}})
    assert r.get("done"), "generation failed"
    return v, f"served + generated with {model}"


def probe_ollabridge_gateway(c, model: str):
    env = {**os.environ, "OLLABRIDGE_API_KEY": "sk-fitlab-ci"}
    subprocess.Popen(["ollabridge", "start", "--no-browser"], env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert wait_http(GATEWAY + "/models", 120), "gateway not up on :11435"
    from openai import OpenAI                                   # top-10 baseline dep
    client = OpenAI(base_url=GATEWAY, api_key=os.environ.get("OLLABRIDGE_API_KEY", "sk-fitlab-ci"))
    models = [m.id for m in client.models.list().data]
    out = client.chat.completions.create(model=model, max_tokens=8,
                                         messages=[{"role": "user", "content": "Say OK"}])
    assert out.choices[0].message.content, "empty completion via gateway"
    return ver("ollabridge"), f"{len(models)} models visible; chat.completions ok"


def probe_agent_via_gateway(c, model: str):
    fid = c["id"]
    common = dict(base_url=GATEWAY, api_key="sk-fitlab-ci")
    if fid in ("langgraph", "langchain"):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, max_tokens=8, **common)
        if fid == "langgraph":
            from langgraph.graph import StateGraph, MessagesState, START, END
            g = StateGraph(MessagesState)
            g.add_node("chat", lambda s: {"messages": [llm.invoke(s["messages"])]})
            g.add_edge(START, "chat"); g.add_edge("chat", END)
            res = g.compile().invoke({"messages": [("user", "Say OK")]})
            assert res["messages"][-1].content
        else:
            assert llm.invoke("Say OK").content
    elif fid == "crewai":
        from crewai import Agent, Crew, Task, LLM
        llm = LLM(model=f"openai/{model}", base_url=GATEWAY, api_key="sk-fitlab-ci", max_tokens=16)
        a = Agent(role="probe", goal="reply OK", backstory="CI probe", llm=llm)
        out = Crew(agents=[a], tasks=[Task(description="Say OK", expected_output="OK", agent=a)]).kickoff()
        assert str(out)
    elif fid == "deepagents":
        from deepagents import create_deep_agent
        from langchain_openai import ChatOpenAI
        agent = create_deep_agent(tools=[], model=ChatOpenAI(model=model, max_tokens=16, **common))
        res = agent.invoke({"messages": [{"role": "user", "content": "Say OK"}]})
        assert res["messages"][-1].content
    else:
        raise ValueError(fid)
    return ver(c["package"]), "one real agent call through OllaBridge → Ollama"


def probe_tracing_decorator(c, model: str):
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-ci"); os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-ci")
    from langfuse import observe

    @observe()
    def traced():
        from openai import OpenAI
        cl = OpenAI(base_url=GATEWAY, api_key="sk-fitlab-ci")
        return cl.chat.completions.create(model=model, max_tokens=5,
                                          messages=[{"role": "user", "content": "OK?"}]).choices[0].message.content
    assert traced()
    return ver("langfuse"), "@observe wrapped a gateway call (offline SDK check)"


def probe_requirements_resolve(c, model: str):
    url = f"https://raw.githubusercontent.com/{c['repo']}/master/requirements.txt"
    p = subprocess.run(f"curl -fsSL {url} -o /tmp/{c['id']}.txt && "
                       f"pip install --dry-run --quiet -r /tmp/{c['id']}.txt", shell=True, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-400:]
    return "latest", "requirements resolve against this week's stack"


PROBES = {"import_only": probe_import_only, "ollama_serve": probe_ollama_serve,
          "ollabridge_gateway": probe_ollabridge_gateway, "agent_via_gateway": probe_agent_via_gateway,
          "tracing_decorator": probe_tracing_decorator, "requirements_resolve": probe_requirements_resolve}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="data/compat_report.json")
    ap.add_argument("--only", nargs="*", help="component ids to run (default: all enabled)")
    args = ap.parse_args()

    stack = yaml.safe_load((ROOT / "data/stack.yaml").read_text())
    model = stack["probe_model"]["ollama_tag"]
    report, failed = {}, 0

    components = [c for c in stack["components"] if c.get("enabled", True)]
    # deterministic order: runtime → gateway → the rest (later probes depend on earlier services)
    order = {"runtime": 0, "gateway": 1}
    components.sort(key=lambda c: order.get(c["role"], 2))
    if args.only:
        components = [c for c in components if c["id"] in args.only]

    for c in components:
        fn = PROBES[c["probe"]]
        try:
            v, note = (fn(c, model) if fn is not probe_import_only else fn(c))
            report[c["id"]] = {"ok": True, "component_version": v, "checked_at": TODAY,
                               "run_url": os.environ.get("RUN_URL", ""), "note": note}
            print(f"✅ {c['name']:<16} {v:<12} {note}")
        except Exception as e:                                   # noqa: BLE001
            report[c["id"]] = {"ok": False, "component_version": ver(c.get("package", "")),
                               "checked_at": TODAY, "run_url": os.environ.get("RUN_URL", ""),
                               "note": f"{type(e).__name__}: {e}"}
            print(f"❌ {c['name']:<16} {report[c['id']]['note']}")
            traceback.print_exc(limit=1)
            if c["role"] != "baseline":
                failed += 1

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({"date": TODAY, "probe_model": model, "results": report}, indent=2))
    print(f"\n→ {args.report} · {failed} non-baseline failures")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
