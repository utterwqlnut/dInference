#!/usr/bin/env bash
# dInference local end-to-end — one script, all processes.
#
# Prereqs:
#   - solana-test-validator running:  solana-test-validator --reset
#   - program deployed (first time only, see `./scripts/local_e2e.sh deploy`)
#   - conda env `dInference` with pip deps installed
#
# Usage:
#   ./scripts/local_e2e.sh up                   # setup + spawn provider + 3 verifiers in background
#   ./scripts/local_e2e.sh down                 # kill all daemons
#   ./scripts/local_e2e.sh status               # pids + chain state
#   ./scripts/local_e2e.sh logs <role>          # tail a log (provider, verifier1-3)
#   ./scripts/local_e2e.sh bounty "prompt"      # post a bounty, wait for response
#   ./scripts/local_e2e.sh challenge <pda>      # challenge a bounty
#   ./scripts/local_e2e.sh pool                 # inspect verifier pool
#   ./scripts/local_e2e.sh deploy               # (re)deploy the program (needs admin key)
#   ./scripts/local_e2e.sh clean                # down + wipe keypairs + wipe logs
#   ./scripts/local_e2e.sh e2e "prompt"         # up → bounty → down (full oneshot)

set -euo pipefail

# ---------- config ----------
MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
HIDDEN_DIM=576
PROGRAM_ID="EDX9NGUiJGtGtLDTHbRCqLrb6Mbf7ES4eKuxumpfUyWq"
RPC_URL="http://localhost:8899"
WS_URL="ws://localhost:8900"

PROVIDER_STAKE=50000000000  # 50 SOL — survives multiple slashes above the 10 SOL min
VERIFIER_STAKE=1000000000
BOUNTY_LAMPORTS=10000000
BOND_LAMPORTS=30000

KEYS_DIR="$HOME/.config/solana"
ADMIN_KEY="$KEYS_DIR/dinference-dev.json"
PROVIDER_KEY="$KEYS_DIR/provider1.json"
VERIFIER_KEYS=("$KEYS_DIR/verifier1.json" "$KEYS_DIR/verifier2.json" "$KEYS_DIR/verifier3.json")
USER_KEY="$KEYS_DIR/user1.json"

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
RUN_DIR="$REPO_ROOT/.run"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"
export SOLANA_RPC_URL="$RPC_URL"
export SOLANA_WS_URL="$WS_URL"
export DINFERENCE_PROGRAM_ID="$PROGRAM_ID"

# ---------- helpers ----------

_mkkey() {
  local path="$1"
  [ -f "$path" ] || solana-keygen new -o "$path" --no-bip39-passphrase --force >/dev/null
}

_airdrop() {
  solana airdrop "${2:-10}" --keypair "$1" --url "$RPC_URL" >/dev/null 2>&1 || true
}

_spawn() {
  local name="$1"; local keypair="$2"; shift 2
  local pidfile="$RUN_DIR/$name.pid"
  local logfile="$LOG_DIR/$name.log"
  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "  $name already running (pid $(cat "$pidfile"))"
    return
  fi
  echo "  starting $name → $logfile"
  (
    cd "$REPO_ROOT"
    DINFERENCE_KEYPAIR_PATH="$keypair" nohup python "$@" >"$logfile" 2>&1 &
    echo $! > "$pidfile"
  )
  sleep 0.5
  if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "  !! $name failed to start — check $logfile" >&2
    rm -f "$pidfile"
    return 1
  fi
  echo "  $name pid $(cat "$pidfile")"
}

_kill() {
  local name="$1"
  local pidfile="$RUN_DIR/$name.pid"
  if [ -f "$pidfile" ]; then
    local pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      echo "  stopped $name (pid $pid)"
    fi
    rm -f "$pidfile"
  fi
}

# ---------- commands ----------

cmd_deploy() {
  _mkkey "$ADMIN_KEY"
  _airdrop "$ADMIN_KEY" 100
  echo "deploying program $PROGRAM_ID ..."
  solana program deploy "$REPO_ROOT/target/deploy/dinference.so" \
      --program-id "$REPO_ROOT/target/deploy/dinference-keypair.json" \
      --upgrade-authority "$ADMIN_KEY" \
      --keypair "$ADMIN_KEY" \
      --url "$RPC_URL"
}

cmd_setup() {
  echo "=== keypairs + airdrops ==="
  for k in "$ADMIN_KEY" "$PROVIDER_KEY" "${VERIFIER_KEYS[@]}" "$USER_KEY"; do
    _mkkey "$k"; _airdrop "$k" 100
    printf "  %-30s %s   bal=%s\n" "$(basename "$k")" "$(solana address -k "$k")" "$(solana balance -k "$k" --url "$RPC_URL" 2>/dev/null || echo ?)"
  done

  echo ""
  echo "=== init_config (idempotent) ==="
  DINFERENCE_KEYPAIR_PATH="$ADMIN_KEY" python - 2>&1 <<'PY' | sed 's/^/  /' || true
import asyncio
from chain import DInferenceClient
async def main():
    c = DInferenceClient()
    try:
        try:
            sig = await c.init_config({
                'min_provider_stake':      10_000_000_000,
                'min_verifier_stake':       1_000_000_000,
                'min_bounty_lamports':          100_000,
                'verify_fee_lamports':           10_000,
                'dishonesty_fee_lamports':    1_000_000,
                'challenger_reward_bps':         12_500,
                'challenge_window_slots':           200,
                'vote_window_slots':                 50,
                'unstake_cooldown_slots':             0,
                'seed_rotation_slots':            1_000,
                'unclaim_timeout_slots':            500,
                'cos_threshold':                  9_900,
                'verifier_committee_size':            3,
                'quorum_fraction':                6_700,
                'max_new_tokens':                    64,
                'initial_seed':                 b'\x42' * 32,
            })
            print('init_config:', sig)
        except Exception as e:
            print('init_config: (already init\'d)')
    finally:
        await c.close()
asyncio.run(main())
PY

  echo ""
  echo "=== init_model (idempotent) ==="
  DINFERENCE_KEYPAIR_PATH="$ADMIN_KEY" MODEL="$MODEL" HIDDEN_DIM="$HIDDEN_DIM" python - 2>&1 <<'PY' | sed 's/^/  /' || true
import asyncio, os
from chain import DInferenceClient
async def main():
    c = DInferenceClient()
    try:
        try:
            sig = await c.init_model(os.environ['MODEL'], int(os.environ['HIDDEN_DIM']))
            print('init_model:', sig)
        except Exception as e:
            print('init_model: (already init\'d)')
    finally:
        await c.close()
asyncio.run(main())
PY

  echo ""
  echo "=== register provider ==="
  DINFERENCE_KEYPAIR_PATH="$PROVIDER_KEY" python -m off_chain_inference.registry \
      --model "$MODEL" --stake $PROVIDER_STAKE 2>&1 | sed 's/^/  /' || echo "  (already registered)"

  echo ""
  echo "=== register verifiers ==="
  for k in "${VERIFIER_KEYS[@]}"; do
    DINFERENCE_KEYPAIR_PATH="$k" python -m off_chain_verifier.registry \
        --model "$MODEL" --stake $VERIFIER_STAKE 2>&1 | sed 's/^/  /' || echo "  $(basename "$k"): (already registered)"
  done

  echo ""
  cmd_pool
}

cmd_up() {
  cmd_setup
  echo ""
  echo "=== spawning daemons ==="
  _spawn provider  "$PROVIDER_KEY"       -m off_chain_inference.ramp --model "$MODEL"
  for i in 1 2 3; do
    _spawn "verifier$i" "${VERIFIER_KEYS[$((i-1))]}" -m off_chain_verifier.ramp --model "$MODEL"
  done
  echo ""
  cmd_status
  echo ""
  echo "logs: tail -f $LOG_DIR/*.log  (or ./scripts/local_e2e.sh logs <role>)"
}

cmd_down() {
  echo "=== stopping daemons ==="
  _kill provider
  _kill verifier1
  _kill verifier2
  _kill verifier3
}

cmd_status() {
  echo "=== status ==="
  for name in provider verifier1 verifier2 verifier3; do
    local pidfile="$RUN_DIR/$name.pid"
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      printf "  %-10s running (pid %s)\n" "$name" "$(cat "$pidfile")"
    else
      printf "  %-10s stopped\n" "$name"
    fi
  done
}

cmd_logs() {
  local role="${1:-}"
  [ -z "$role" ] && { echo "usage: $0 logs <provider|verifier1|verifier2|verifier3>"; exit 1; }
  local file="$LOG_DIR/$role.log"
  [ -f "$file" ] || { echo "no log file at $file"; exit 1; }
  tail -f "$file"
}

cmd_pool() {
  MODEL="$MODEL" python - 2>&1 <<'PY' | sed 's/^/  /'
import asyncio, os
from chain import DInferenceClient
from solders.pubkey import Pubkey
async def main():
    c = DInferenceClient()
    try:
        pool = await c.fetch_pool(os.environ['MODEL'])
        if pool is None:
            print('pool: not initialized')
            return
        print(f'pool size: {len(pool.available)}')
        for pk in pool.available:
            print(f'  - {Pubkey.from_bytes(pk)}')
    finally:
        await c.close()
asyncio.run(main())
PY
}

cmd_bounty() {
  local prompt="${*:-What is 2+2?}"
  DINFERENCE_KEYPAIR_PATH="$USER_KEY" PROMPT="$prompt" MODEL="$MODEL" BOUNTY="$BOUNTY_LAMPORTS" python - <<'PY'
import asyncio, os
from sdk import DInference
async def main():
    d = DInference()
    try:
        r = await d.inference(
            prompt=os.environ['PROMPT'],
            model_id=os.environ['MODEL'],
            bounty_lamports=int(os.environ['BOUNTY']),
            timeout_s=120,
        )
        print('bounty_pda:', r.bounty_pda)
        print('---')
        print(r.response_text if r.response_text else '(timed out — check provider logs)')
    finally:
        await d.close()
asyncio.run(main())
PY
}

cmd_challenge() {
  local b="${1:-}"
  [ -z "$b" ] && { echo "usage: $0 challenge <bounty_pda>"; exit 1; }
  DINFERENCE_KEYPAIR_PATH="$USER_KEY" BOUNTY="$b" MODEL="$MODEL" BOND="$BOND_LAMPORTS" python - <<'PY'
import asyncio, os
from sdk import DInference
from solders.pubkey import Pubkey
async def main():
    d = DInference()
    try:
        sig = await d.challenge(
            Pubkey.from_string(os.environ['BOUNTY']),
            model_id=os.environ['MODEL'],
            bond_lamports=int(os.environ['BOND']),
        )
        print('challenge tx:', sig)
    finally:
        await d.close()
asyncio.run(main())
PY
}

cmd_clean() {
  cmd_down
  echo "=== wiping keypairs (except admin) + logs ==="
  rm -f "$PROVIDER_KEY" "${VERIFIER_KEYS[@]}" "$USER_KEY"
  rm -rf "$LOG_DIR"/* "$RUN_DIR"/*
  echo "done. restart validator (solana-test-validator --reset) to also wipe on-chain state."
}

cmd_e2e() {
  cmd_up
  echo ""
  echo "waiting 5s for daemons to subscribe to events..."
  sleep 5
  echo ""
  echo "=== posting bounty ==="
  cmd_bounty "${*:-What is 2+2?}"
  echo ""
  echo "(daemons still running — call './scripts/local_e2e.sh down' when done)"
}

# ---------- dispatch ----------

case "${1:-}" in
  up)         shift; cmd_up         "$@" ;;
  down)       shift; cmd_down       "$@" ;;
  status)     shift; cmd_status     "$@" ;;
  logs)       shift; cmd_logs       "$@" ;;
  bounty)     shift; cmd_bounty     "$@" ;;
  challenge)  shift; cmd_challenge  "$@" ;;
  pool)       shift; cmd_pool       "$@" ;;
  setup)      shift; cmd_setup      "$@" ;;
  deploy)     shift; cmd_deploy     "$@" ;;
  clean)      shift; cmd_clean      "$@" ;;
  e2e)        shift; cmd_e2e        "$@" ;;
  *)
    cat <<EOF
Usage: $0 <command> [args]

Lifecycle:
  deploy                    one-time: deploy program (needs admin keypair + funded)
  up                        setup + spawn provider + 3 verifiers (background)
  down                      stop all daemons
  clean                     down + wipe keypairs + wipe logs
  e2e "prompt"              up + bounty + leave daemons running (full happy-path oneshot)

Observability:
  status                    daemon pids + state
  logs <role>               tail -f one role (provider, verifier1-3)
  pool                      show verifier pool

Actions:
  bounty "prompt"           post a bounty, wait for response
  challenge <bounty_pda>    challenge a bounty

Keypairs under $KEYS_DIR; logs under $LOG_DIR; pidfiles under $RUN_DIR.
Requires solana-test-validator running on $RPC_URL.
EOF
    exit 1
    ;;
esac
