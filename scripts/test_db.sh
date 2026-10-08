#!/usr/bin/env bash
# Applies supabase/migrations to a throwaway Postgres (with Supabase stubs)
# and runs the pgTAP tests in supabase/tests/database.
#
# Requires PostgreSQL (initdb, pg_ctl, psql) with the pgTAP extension, e.g.
#   apt-get install postgresql-16 postgresql-16-pgtap
# With Docker available, `npx supabase start && npx supabase test db` runs the
# same tests against a full local Supabase instead.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PG_BIN="${PG_BIN:-$(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | sort -V | tail -1)}"
export PATH="$PG_BIN:$PATH"

WORK="$(mktemp -d)"
PORT="${PGPORT_TEST:-54329}"
cleanup() { pg_ctl -D "$WORK/data" -m immediate stop >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT

if [ "$(id -u)" = "0" ]; then
  # Postgres refuses to run as root: run the throwaway cluster as 'postgres'
  chown -R postgres "$WORK"
  RUN=(runuser -u postgres --)
else
  RUN=()
fi

"${RUN[@]}" initdb -D "$WORK/data" -U postgres -A trust >/dev/null
"${RUN[@]}" pg_ctl -D "$WORK/data" -o "-p $PORT -k $WORK -c listen_addresses=''" -l "$WORK/pg.log" -w start >/dev/null

PSQL=(psql -h "$WORK" -p "$PORT" -U postgres -d postgres -v ON_ERROR_STOP=1 -q -X)

"${PSQL[@]}" -f "$ROOT/scripts/db/supabase_stubs.sql"
for migration in "$ROOT"/supabase/migrations/*.sql; do
  echo "applying $(basename "$migration")"
  "${PSQL[@]}" -f "$migration"
done

status=0
for test in "$ROOT"/supabase/tests/database/*.sql; do
  echo "== $(basename "$test")"
  output="$("${PSQL[@]}" -t -A -f "$test" 2>&1)" || status=1
  echo "$output"
  if grep -qE '^not ok|Looks like|ERROR' <<<"$output"; then status=1; fi
done
exit $status
