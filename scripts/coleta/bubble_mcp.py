"""Minimal read-only client for the bubble-bridge MCP configured in opencode (headers never printed).

Usage: python bubble_mcp.py <tool> '<json args>'   (only READ_TOOLS are allowed)
"""
import json
import os
import sys
import urllib.request

READ_TOOLS = {"bubble_config", "bubble_schema", "bubble_query", "bubble_design_read"}
CONFIG = os.path.expanduser("~/.config/opencode/opencode.json")
PROTOCOL = "2025-03-26"


class Client:
    def __init__(self) -> None:
        server = json.load(open(CONFIG, encoding="utf-8"))["mcp"]["bubble"]
        self.url, self.headers = server["url"], dict(server.get("headers") or {})
        self.session, self.next_id = None, 0
        self.rpc("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                "clientInfo": {"name": "laya-cypress", "version": "1"}})
        self.rpc("notifications/initialized", notify=True)

    def rpc(self, method: str, params: dict | None = None, notify: bool = False) -> dict | None:
        body: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            body["params"] = params
        if not notify:
            self.next_id += 1
            body["id"] = self.next_id
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **self.headers}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        request = urllib.request.Request(self.url, json.dumps(body).encode(), headers, method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            self.session = response.headers.get("Mcp-Session-Id") or self.session
            raw = response.read().decode()
        if notify or not raw.strip():
            return None
        if raw.lstrip().startswith("{"):
            return json.loads(raw)
        return json.loads([line[5:].strip() for line in raw.splitlines() if line.startswith("data:")][-1])

    def tool(self, name: str, arguments: dict) -> dict:
        if name not in READ_TOOLS:
            raise ValueError(f"tool fora da lista de leitura: {name}")
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})["result"]
        texts = [item["text"] for item in result.get("content", []) if item.get("type") == "text"]
        try:
            return json.loads(texts[0]) if len(texts) == 1 else {"textos": texts}
        except json.JSONDecodeError:
            return {"texto": texts[0]}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(Client().tool(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}),
                     ensure_ascii=False, indent=1)[:12000])
