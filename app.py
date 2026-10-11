"""Catalog Agent web console.

    python app.py            # http://127.0.0.1:8000
    python app.py --port 8080 --host 0.0.0.0
"""
import argparse

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Catalog Agent web console.")
    p.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to allow other machines on the network")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    print(f"Catalog Agent console on http://{args.host}:{args.port}")
    uvicorn.run("webapp.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
