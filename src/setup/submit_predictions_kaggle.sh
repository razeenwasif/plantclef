#!/usr/bin/env bash
set -Eeuo pipefail

COMP="plantclef-2026"
INPUT_DIR="."
PATTERN="submission_*.csv"
OUT="scores.csv"
POLL_INTERVAL=20
WAIT_TIMEOUT=3600
SUBMIT_RETRIES=3
RETRY_DELAY=10
FORCE=0

log() {
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
}

die() {
    log "ERROR: $*"
    exit 1
}

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  -c, --competition NAME     Kaggle competition name (default: $COMP)
  -i, --input-dir DIR        Directory containing CSVs (default: $INPUT_DIR)
  -p, --pattern GLOB         File pattern to match (default: $PATTERN)
  -o, --output FILE          Output CSV file (default: $OUT)
      --poll-seconds N       Poll interval in seconds (default: $POLL_INTERVAL)
      --timeout-seconds N    Max wait per submission (default: $WAIT_TIMEOUT)
      --submit-retries N     Submit retries (default: $SUBMIT_RETRIES)
      --retry-delay N        Delay between retries (default: $RETRY_DELAY)
      --force                Resubmit files already recorded in output
  -h, --help                 Show this help
EOF
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

csv_escape() {
    local s="$1"
    s=${s//\"/\"\"}
    printf '"%s"' "$s"
}

append_csv_row() {
    local file="$1"
    local message="$2"
    local status="$3"
    local public_score="$4"

    {
        csv_escape "$file"; printf ','
        csv_escape "$message"; printf ','
        csv_escape "$status"; printf ','
        csv_escape "$public_score"; printf '\n'
    } >> "$OUT"
}

ensure_output_file() {
    if [[ ! -f "$OUT" ]]; then
        printf '"file","message","status","public_score"\n' > "$OUT"
    fi
}

already_recorded() {
    local desc_key="$1"
    [[ -f "$OUT" ]] || return 1

    awk -F',' -v target="$desc_key" '
        NR == 1 { next }
        {
            msg = $2
            gsub(/\r/, "", msg)
            gsub(/^"/, "", msg)
            gsub(/"$/, "", msg)
            gsub(/""/, "\"", msg)

            # Match either exact key or key followed by timestamp suffix
            if (msg == target || index(msg, target "__") == 1) {
                found = 1
                exit
            }
        }
        END { exit(found ? 0 : 1) }
    ' "$OUT"
}

list_files() {
    find "$INPUT_DIR" -type f -name "$PATTERN" -print0 | sort -zV
}

submit_with_retries() {
    local file="$1"
    local message="$2"
    local attempt output

    for (( attempt=1; attempt<=SUBMIT_RETRIES; attempt++ )); do
        log "Submitting $(basename "$file") (attempt $attempt/$SUBMIT_RETRIES)"

        if output="$(kaggle competitions submit -c "$COMP" -f "$file" -m "$message" 2>&1)"; then
            log "$output"
            return 0
        fi

        log "Submit failed for $(basename "$file"): $output"

        if (( attempt < SUBMIT_RETRIES )); then
            sleep "$RETRY_DELAY"
        fi
    done

    return 1
}

fetch_submission_row() {
    local message="$1"

    kaggle competitions submissions -c "$COMP" -v -q | awk -F',' -v msg="$message" '
        function clean(s) {
            gsub(/\r/, "", s)
            gsub(/^"/, "", s)
            gsub(/"$/, "", s)
            gsub(/""/, "\"", s)
            return s
        }
        function keyify(s) {
            s = tolower(clean(s))
            gsub(/[^a-z0-9]/, "", s)
            return s
        }
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                header[keyify($i)] = i
            }
            next
        }
        {
            msg_idx  = header["description"] ? header["description"] : header["message"]
            stat_idx = header["status"]
            pub_idx  = header["publicscore"] ? header["publicscore"] : header["score"]

            if (!msg_idx || !stat_idx) {
                next
            }

            desc = clean($(msg_idx))
            if (desc == msg) {
                status = clean($(stat_idx))
                public_score = pub_idx ? clean($(pub_idx)) : ""
                print status "," public_score
                found = 1
                exit
            }
        }
        END {
            if (!found) {
                print "NOT_FOUND,"
            }
        }
    '
}

wait_for_score() {
    local file="$1"
    local message="$2"
    local deadline=$((SECONDS + WAIT_TIMEOUT))
    local last_status=""

    while (( SECONDS < deadline )); do
        local parsed
        if ! parsed="$(fetch_submission_row "$message")"; then
            log "Could not fetch submissions for $(basename "$file"); retrying"
            sleep "$POLL_INTERVAL"
            continue
        fi

        local status public_score
        IFS=',' read -r status public_score <<< "$parsed"

        if [[ "$status" == "NOT_FOUND" ]]; then
            log "Waiting for $(basename "$file") to appear..."
            sleep "$POLL_INTERVAL"
            continue
        fi

        local status_lc
        status_lc="$(printf '%s' "$status" | tr '[:upper:]' '[:lower:]')"

        if [[ "$status_lc" != "$last_status" ]]; then
            log "$(basename "$file") status=$status public=$public_score"
            last_status="$status_lc"
        fi

        case "$status_lc" in
            *complete|*error|failed)
                append_csv_row "$(basename "$file")" "$message" "$status" "$public_score"
                return 0
                ;;
        esac

        sleep "$POLL_INTERVAL"
    done

    log "Timed out waiting for $(basename "$file")"
    append_csv_row "$(basename "$file")" "$message" "timeout" ""
    return 1
}

pretty_print_csv() {
    awk -F',' '
        function clean(s) {
            gsub(/^"/, "", s)
            gsub(/"$/, "", s)
            gsub(/""/, "\"", s)
            return s
        }
        {
            for (i = 1; i <= NF; i++) {
                $i = clean($i)
            }
            OFS = "\t"
            print $1, $2, $3, $4
        }
    ' "$OUT" | column -s $'\t' -t
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c|--competition)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                COMP="$2"
                shift 2
                ;;
            -i|--input-dir)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                INPUT_DIR="$2"
                shift 2
                ;;
            -p|--pattern)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                PATTERN="$2"
                shift 2
                ;;
            -o|--output)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                OUT="$2"
                shift 2
                ;;
            --poll-seconds)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                POLL_INTERVAL="$2"
                shift 2
                ;;
            --timeout-seconds)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                WAIT_TIMEOUT="$2"
                shift 2
                ;;
            --submit-retries)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                SUBMIT_RETRIES="$2"
                shift 2
                ;;
            --retry-delay)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                RETRY_DELAY="$2"
                shift 2
                ;;
            --force)
                FORCE=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "Unknown argument: $1"
                ;;
        esac
    done

    require_cmd kaggle
    require_cmd awk
    require_cmd find
    require_cmd sort

    [[ -d "$INPUT_DIR" ]] || die "Input directory does not exist: $INPUT_DIR"

    ensure_output_file

    local -a files=()
    while IFS= read -r -d '' f; do
        files+=("$f")
    done < <(list_files)

    (( ${#files[@]} > 0 )) || die "No files matched '$PATTERN' in $INPUT_DIR"

    log "Competition: $COMP"
    log "Input dir:    $INPUT_DIR"
    log "Pattern:      $PATTERN"
    log "Output:       $OUT"
    log "Files found:  ${#files[@]}"

    local failures=0

    for file in "${files[@]}"; do
        local base abs_path safe_path message desc_key
        base="$(basename "$file")"

        abs_path="$(realpath "$file")"

        safe_path="${abs_path//\//__}"
        safe_path="${safe_path//[^A-Za-z0-9_.-]/_}"

        desc_key="$safe_path"
        message="${desc_key}__$(date -u +%Y%m%dT%H%M%SZ)"

        if (( FORCE == 0 )) && already_recorded "$desc_key"; then
            log "Skipping already recorded description: $desc_key"
            continue
        fi

        if ! submit_with_retries "$file" "$message"; then
            log "Giving up on $base after submit failures"
            append_csv_row "$base" "$message" "submit_failed" ""
            ((failures++))
            continue
        fi

        if ! wait_for_score "$file" "$message"; then
            ((failures++))
        fi
    done

    echo
    log "Done. Results saved in $OUT"

    if command -v column >/dev/null 2>&1; then
        pretty_print_csv
    else
        cat "$OUT"
    fi

    (( failures == 0 ))
}

main "$@"
