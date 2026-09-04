import asyncio
import json
import websockets

async def main():
    uri = "ws://127.0.0.1:22400/ws/duplex?session_id=test_expert_ws_01"
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        print("Connected! Sending prepare with expert_config...")
        await ws.send(json.dumps({
            "type": "prepare",
            "system_prompt": "Eres un asistente en espanol.",
            "expert_config": {
                "provider": "agy",
                "enabled": True,
            }
        }))

        while True:
            resp = json.loads(await ws.recv())
            print("Received:", resp.get("type"))
            if resp.get("type") == "prepared":
                break

        print("\n--- Test 1: Explicit ask_expert query (AGY) ---")
        await ws.send(json.dumps({
            "type": "ask_expert",
            "query": "Explica que es la gravedad segun Einstein en dos oraciones.",
            "provider": "agy",
        }))

        thinking_received = False
        done_received = False

        while not done_received:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30.0))
            if msg.get("type") == "expert_status":
                status = msg.get("status")
                print(f"[ExpertStatus] status={status}, prov={msg.get('provider')}")
                if status == "thinking":
                    thinking_received = True
                elif status == "done":
                    done_received = True
                    print(f"Expert Response: {msg.get('text')}")
                    print(f"Latency: {msg.get('elapsed_ms')}ms")

        assert thinking_received, "Expected thinking status"
        assert done_received, "Expected done status"
        print("\nTest 1 PASSED!")

        print("\n--- Test 2: Claude provider ask_expert query ---")
        await ws.send(json.dumps({
            "type": "ask_expert",
            "query": "Cual es la distancia a la luna en kilometros en una frase corta.",
            "provider": "claude",
        }))

        claude_done = False
        while not claude_done:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30.0))
            if msg.get("type") == "expert_status":
                status = msg.get("status")
                print(f"[Claude ExpertStatus] status={status}, prov={msg.get('provider')}")
                if status == "done":
                    claude_done = True
                    print(f"Claude Response: {msg.get('text')}")

        assert claude_done, "Expected claude done status"
        print("\nTest 2 PASSED!")

        await ws.send(json.dumps({"type": "stop"}))
        print("Session stopped cleanly.")

if __name__ == "__main__":
    asyncio.run(main())
