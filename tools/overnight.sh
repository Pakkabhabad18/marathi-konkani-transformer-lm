#!/bin/bash
#
# Overnight collection supervisor.
#
# WHY THIS EXISTS
# ---------------
# Collectors exit for reasons that are not failures: a site's sitemaps get
# exhausted, a network blip kills a run, a streamed dataset ends. When that
# happens at 2am the job simply stops and hours of collection time are lost.
#
# Every collector in this project is checkpointed and resumable, and skips URLs
# and identifiers it has already seen. So restarting one is cheap and safe - it
# picks up where it stopped and re-fetches nothing. That makes "restart it
# whenever it exits" the correct policy, and this script applies it.
#
# It also appends a health snapshot every 30 minutes, so in the morning there is
# a timeline of progress rather than a single instant, and a stall is visible
# after the fact.
#
# USAGE
#   bash tools/overnight.sh start     # start everything under supervision
#   bash tools/overnight.sh status    # what is running right now
#   bash tools/overnight.sh stop      # stop supervisors and collectors
#
# Logs land in logs/. The supervisor writes logs/supervisor.log and
# logs/health_timeline.log.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

mkdir -p logs .run

PIDFILE=".run/supervisors.pid"
RESTART_DELAY=60
HEALTH_EVERY=1800          # 30 minutes

# Each entry: name|command
JOBS=(
  "marathi_news|python3 marathi/scripts/collect_news.py --workers 8"
  "marathi_archive_gr|python3 marathi/scripts/collect_archive_gr.py --workers 12"
  "konkani_books|python3 konkani/scripts/ingest_books_corpus.py"
  "konkani_news|python3 konkani/scripts/collect_news.py --workers 6"
)

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

supervise() {
  local name="$1"
  local cmd="$2"
  local log="logs/${name}.log"

  while true; do
    echo "[$(stamp)] SUPERVISOR: starting ${name}" >> "$log"
    # shellcheck disable=SC2086
    $cmd >> "$log" 2>&1
    local rc=$?
    echo "[$(stamp)] SUPERVISOR: ${name} exited rc=${rc}; restarting in ${RESTART_DELAY}s" >> "$log"

    # A collector that exits instantly, repeatedly, is broken rather than
    # finished. Back off so a broken job does not spin the CPU all night.
    sleep "$RESTART_DELAY"
  done
}

start() {
  if [[ -f "$PIDFILE" ]] && pgrep -qF "$PIDFILE" 2>/dev/null; then
    echo "Supervisors already running. Use 'stop' first."
    exit 1
  fi
  : > "$PIDFILE"

  echo "Starting supervised collection at $(stamp)"
  echo "[$(stamp)] ===== supervisor start =====" >> logs/supervisor.log

  for entry in "${JOBS[@]}"; do
    local name="${entry%%|*}"
    local cmd="${entry#*|}"
    supervise "$name" "$cmd" &
    echo $! >> "$PIDFILE"
    echo "  ${name}  supervisor pid $!"
    echo "[$(stamp)] supervising ${name} (pid $!): ${cmd}" >> logs/supervisor.log
  done

  # Health timeline.
  (
    while true; do
      {
        echo ""
        echo "########## $(stamp) ##########"
        python3 tools/health_check.py --all 2>&1
      } >> logs/health_timeline.log
      sleep "$HEALTH_EVERY"
    done
  ) &
  echo $! >> "$PIDFILE"
  echo "  health monitor  pid $! (every $((HEALTH_EVERY / 60)) min -> logs/health_timeline.log)"

  # Keep the Mac awake. Without this the laptop sleeps and everything pauses.
  if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -s &
    echo $! >> "$PIDFILE"
    echo "  caffeinate      pid $! (prevents sleep)"
  fi

  echo ""
  echo "All jobs supervised. They will restart automatically whenever they exit."
  echo "Morning check:  bash tools/overnight.sh status"
  echo "Stop:           bash tools/overnight.sh stop"
}

status() {
  echo "=== supervisors ==="
  if [[ -f "$PIDFILE" ]]; then
    while read -r pid; do
      if kill -0 "$pid" 2>/dev/null; then
        echo "  pid $pid  RUNNING"
      else
        echo "  pid $pid  gone"
      fi
    done < "$PIDFILE"
  else
    echo "  none (supervisor not started)"
  fi

  echo ""
  echo "=== collector processes ==="
  pgrep -fl "collect_news.py|collect_archive_gr.py|ingest_books_corpus.py|ingest_indiccorp.py" \
    || echo "  none running"

  echo ""
  echo "=== progress ==="
  python3 tools/health_check.py --all
}

stop() {
  echo "Stopping supervisors and collectors..."
  if [[ -f "$PIDFILE" ]]; then
    while read -r pid; do
      kill "$pid" 2>/dev/null && echo "  killed supervisor $pid"
    done < "$PIDFILE"
    rm -f "$PIDFILE"
  fi
  pkill -f "collect_news.py" 2>/dev/null && echo "  stopped collect_news"
  pkill -f "collect_archive_gr.py" 2>/dev/null && echo "  stopped collect_archive_gr"
  pkill -f "ingest_books_corpus.py" 2>/dev/null && echo "  stopped ingest_books_corpus"
  pkill -f "caffeinate" 2>/dev/null && echo "  stopped caffeinate"
  echo "[$(stamp)] ===== supervisor stop =====" >> logs/supervisor.log
  echo "Done. All checkpoints are saved; restarting resumes where it stopped."
}

case "${1:-}" in
  start)  start ;;
  status) status ;;
  stop)   stop ;;
  *)
    echo "Usage: bash tools/overnight.sh {start|status|stop}"
    exit 1
    ;;
esac
