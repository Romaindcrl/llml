#!/usr/bin/env python3
"""RunPod provisioning CLI for the LLML eval campaign (GraphQL API).

Usage:
  python runpod_api.py balance
  python runpod_api.py deploy [--name llml-eval] [--gpu "NVIDIA GeForce RTX 4090"]
                              [--cloud COMMUNITY] [--disk 40] [--volume 150]
  python runpod_api.py status [POD_ID]
  python runpod_api.py stop   [POD_ID]     # pause billing (GPU released, volume kept)
  python runpod_api.py resume [POD_ID]
  python runpod_api.py terminate [POD_ID]  # destroy pod + volume
  python runpod_api.py pods                # list all pods

State (pod id + exec token) is stored in eval/runpod/.pod.json (gitignored).
Requires: RUNPOD_API_KEY env var.
"""
import argparse, base64, json, os, secrets, sys, time
import urllib.request

API = "https://api.runpod.io/graphql"
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, ".pod.json")


def gql(query, variables=None):
    req = urllib.request.Request(
        API,
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0",
                 "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"], indent=2))
    return out["data"]


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {}


def save_state(st, path=None):
    path = path or STATE
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)
    with open(path, "w") as f:
        json.dump(st, f, indent=2)
    print(f"state -> {path}")


def pod_arg(args):
    pid = getattr(args, "pod_id", None) or load_state().get("pod_id")
    if not pid:
        sys.exit("no pod id (arg or .pod.json)")
    return pid


def cmd_balance(_):
    d = gql("query { myself { clientBalance spendLimit currentSpendPerHr } }")
    print(json.dumps(d["myself"], indent=2))


def cmd_pods(_):
    d = gql("""query { myself { pods { id name desiredStatus costPerHr gpuCount
               machine { gpuDisplayName } runtime { uptimeInSeconds } } } }""")
    print(json.dumps(d["myself"]["pods"], indent=2))


def cmd_deploy(args):
    token = secrets.token_hex(24)
    with open(os.path.join(HERE, "agent_server.py"), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    docker_args = (
        "bash -c 'mkdir -p /workspace && echo " + b64 +
        " | base64 -d > /agent_server.py && exec python3 -u /agent_server.py'"
    )
    mutation = """
    mutation Deploy($input: PodFindAndDeployOnDemandInput) {
      podFindAndDeployOnDemand(input: $input) {
        id imageName machineId costPerHr machine { gpuDisplayName }
      }
    }"""
    inp = {
        "cloudType": args.cloud,
        "gpuCount": 1,
        "gpuTypeId": args.gpu,
        "name": args.name,
        "imageName": args.image,
        "containerDiskInGb": args.disk,
        "volumeInGb": args.volume,
        "volumeMountPath": "/workspace",
        "minVcpuCount": args.min_vcpu,
        "minMemoryInGb": args.min_mem,
        "ports": "8888/http",
        "dockerArgs": docker_args,
        "env": [{"key": "EXEC_TOKEN", "value": token}],
    }
    pod = None
    for attempt in range(args.retries):
        try:
            d = gql(mutation, {"input": inp})
            pod = d["podFindAndDeployOnDemand"]
            if pod:
                break
        except RuntimeError as e:
            print(f"tentative {attempt + 1}/{args.retries} echouee: "
                  f"{str(e)[:120]}", file=sys.stderr)
            time.sleep(5)
    if not pod:
        sys.exit("deploiement impossible apres retries")
    st = {"pod_id": pod["id"], "token": token,
          "url": f"https://{pod['id']}-8888.proxy.runpod.net",
          "cost_per_hr": pod["costPerHr"], "gpu": pod["machine"]["gpuDisplayName"],
          "image": args.image, "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save_state(st, getattr(args, "state", None))
    print(json.dumps(pod, indent=2))


def cmd_status(args):
    pid = pod_arg(args)
    d = gql("""query Pod($id: String!) { pod(input: {podId: $id}) {
        id name desiredStatus costPerHr
        runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } gpus { gpuUtilPercent memoryUtilPercent } }
    } }""", {"id": pid})
    print(json.dumps(d["pod"], indent=2))


def cmd_stop(args):
    pid = pod_arg(args)
    d = gql("mutation Stop($id: String!) { podStop(input: {podId: $id}) { id desiredStatus } }",
            {"id": pid})
    print(json.dumps(d, indent=2))


def cmd_resume(args):
    pid = pod_arg(args)
    d = gql("mutation Resume($id: String!) { podResume(input: {podId: $id, gpuCount: 1}) { id desiredStatus costPerHr } }",
            {"id": pid})
    print(json.dumps(d, indent=2))


def cmd_terminate(args):
    pid = pod_arg(args)
    gql("mutation Kill($id: String!) { podTerminate(input: {podId: $id}) }", {"id": pid})
    print(f"terminated {pid}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance")
    sub.add_parser("pods")
    dp = sub.add_parser("deploy")
    dp.add_argument("--state", default=None,
                    help="fichier d'état de sortie (multi-pods), défaut .pod.json")
    dp.add_argument("--name", default="llml-eval")
    dp.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    dp.add_argument("--cloud", default="COMMUNITY")
    dp.add_argument("--image", default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    dp.add_argument("--disk", type=int, default=40)
    dp.add_argument("--volume", type=int, default=150)
    dp.add_argument("--min-vcpu", type=int, default=6)
    dp.add_argument("--min-mem", type=int, default=24)
    dp.add_argument("--retries", type=int, default=6)
    for name in ("status", "stop", "resume", "terminate"):
        sp = sub.add_parser(name)
        sp.add_argument("pod_id", nargs="?")
    a = p.parse_args()
    {"balance": cmd_balance, "pods": cmd_pods, "deploy": cmd_deploy,
     "status": cmd_status, "stop": cmd_stop, "resume": cmd_resume,
     "terminate": cmd_terminate}[a.cmd](a)
