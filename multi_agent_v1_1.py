#!/usr/bin/env python3
"""
================================================================================
Enterprise Multi-Agent Loop System — v1.1.0 (PATCHED)
Principal Engineer: High-Throughput Agentic Design Patterns
Architecture: Dual-Agent State Machine with Domain Context Triage

PATCH NOTES v1.1:
  - call_groq_api() now returns (bool, str) tuple — surfaces real API errors
  - HTTP 401/429/5xx bodies are printed before any JSON parsing is attempted
  - run_agent_1_triage() separates network failures from JSON parse failures
  - run_agent_2_executive() same separation applied
  - All [ERROR] lines now show root cause: HTTP code, connection issue, or bad JSON
================================================================================

SYSTEM TOPOLOGY:
  [Input Matrix] --> [Agent 1: Strategic Context Triage]
                          |
                    Score >= 7?
                    /         \
                  YES          NO
                   |            |
  [Agent 2: Executive       [Logged &
   Content Engine]           Skipped]
        |
  [global_state.json]

DEPENDENCIES: requests (pip install requests), json, time, re, datetime (stdlib)
================================================================================
"""

import json
import re
import time
from datetime import datetime, timezone

import requests

# ==============================================================================
# SECTION 0: ENVIRONMENT ISOLATION
# Hardcoded runtime key — fully insulated from shell env vars and session
# overrides. Replace the value below with your real Groq API key.
# Get one free at: https://console.groq.com
# ==============================================================================
ACTIVE_KEY = "GROQ_API_KEY"

# ==============================================================================
# SECTION 1: SYSTEM CONSTANTS
# ==============================================================================

# Official Groq OpenAI-compatible inference endpoint
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Active model identifier
MODEL_ID = "llama-3.1-8b-instant"

# Agent scoring threshold — records at or above this score route to Agent 2
HIGH_IMPACT_THRESHOLD = 7

# Rate-limit guard: mandatory pause before each network call
RATE_LIMIT_SLEEP_SECONDS = 10

# Output state file — committed at end of every run
STATE_OUTPUT_FILE = "global_state.json"

# ==============================================================================
# SECTION 2: MULTI-DOMAIN INPUT MATRIX
# Five real-world datasets spanning distinct enterprise verticals.
# Each record is a self-contained context payload for Agent 1.
# ==============================================================================

INPUT_MATRIX = [
    {
        "domain": "Health",
        "record_id": "HLTH-001",
        "payload": (
            "Patient ID 7734-B, 67-year-old male, ICU admission 14:32 UTC. "
            "Vitals: BP 88/52 mmHg (critical hypotension), HR 121 bpm, SpO2 91% on 6L O2, "
            "Temp 39.8 C, GCS 11. Labs: Lactate 4.2 mmol/L, WBC 18.4 x10^9/L, "
            "Procalcitonin 22 ng/mL. Suspected septic shock. Vasopressor initiation pending. "
            "Triage nurse flagged delayed antibiotic administration (>3 hrs). "
            "Hospital bed occupancy at 97%. Nearest ICU transfer option: 42 km."
        ),
    },
    {
        "domain": "Education",
        "record_id": "EDUC-002",
        "payload": (
            "District 14 secondary school cohort, Grade 10, n=340 students. "
            "Mid-term diagnostic results: Mathematics pass rate 38% (target 70%), "
            "Reading comprehension 54%, STEM project completion 29%. "
            "Teacher-to-student ratio: 1:52 (recommended 1:30). "
            "Three curriculum modules (Algebra II, Physics Kinematics, Data Literacy) "
            "show >60% failure rates. Attendance correlation analysis shows 71% of "
            "failing students have >15% absenteeism. Digital device access: 44% of "
            "students lack home internet. Budget allocation for remedial programs: USD 0."
        ),
    },
    {
        "domain": "Entertainment",
        "record_id": "ENTR-003",
        "payload": (
            "Indie artist: Mara Osei, Genre: Afrobeats-Soul fusion. "
            "Streaming data Q3: Spotify 1.2M streams, Apple Music 340K, YouTube 4.7M views. "
            "Royalty disbursement: USD 1,840 (effective per-stream rate: USD 0.0011). "
            "Sync licensing inquiries: 3 pending, 0 converted. "
            "Social engagement: TikTok 22K shares on hero track, Instagram followers +18% MoM. "
            "Distribution contract clause 7.3 restricts playlist pitching for 90 days post-release. "
            "Merchandise revenue: USD 620. Label advance recoupment outstanding: USD 14,500. "
            "Estimated breakeven streams at current rate: 13.2M."
        ),
    },
    {
        "domain": "Sports",
        "record_id": "SPRT-004",
        "payload": (
            "Athlete: J. Kimani, 400m sprinter, 22 yrs. Pre-season biometric report. "
            "VO2 Max: 58.4 mL/kg/min (elite threshold: 62+). Lactate threshold: 14.1 km/h. "
            "HRV 7-day avg: 41 ms (baseline 67 ms, 39% decline, overtraining marker). "
            "Sleep quality score: 4.1/10 (Whoop). Body fat: 8.9%. Muscle symmetry index: "
            "left quad 12% weaker than right. Hamstring tightness: Grade 1 strain flag. "
            "Training load last 14 days: 2.3x above periodized plan. "
            "Upcoming competition: National Championships in 18 days. "
            "Physiotherapist recommendation: mandatory 5-day rest protocol. Coach override: active."
        ),
    },
    {
        "domain": "Politics/Institutions",
        "record_id": "POLI-005",
        "payload": (
            "Public sentiment analysis, National Housing Policy Draft v2.1, "
            "data window: 30 days, n=184,000 social/media signals. "
            "Sentiment breakdown: Positive 18%, Neutral 31%, Negative 51%. "
            "Top negative themes: affordability crisis (34%), displacement fears (22%), "
            "corruption suspicion in land allocation (19%), inadequate rural coverage (25%). "
            "Petition signatures against current draft: 47,200 in 12 days. "
            "Parliamentary debate scheduled in 9 days. Opposition bloc: 6 legislators "
            "have signaled vote against. Governing coalition margin: 4 seats. "
            "Think-tank policy brief cited by 3 major outlets recommending 40% affordable "
            "unit mandate and independent oversight board."
        ),
    },
]

# ==============================================================================
# SECTION 3: UTILITY FUNCTIONS
# ==============================================================================


def build_headers() -> dict:
    """
    Construct authenticated HTTP headers for Groq API requests.

    Returns:
        dict: HTTP headers with Bearer token and content type.
    """
    return {
        "Authorization": f"Bearer {ACTIVE_KEY}",
        "Content-Type": "application/json",
    }


def safe_json_parse(raw_text: str) -> dict:
    """
    Robust JSON recovery parser with explicit string-slicing fallback.

    Strategy (in order):
      1. Direct json.loads() on the raw response.
      2. Strip triple-backtick markdown fences (```json ... ```) via regex,
         then retry json.loads().
      3. String-slice fallback: locate the first '{' and last '}' and extract
         the substring, then retry json.loads().
      4. If all strategies fail, return a structured error dict — never crash.

    Args:
        raw_text (str): Raw string content from the LLM response.

    Returns:
        dict: Parsed JSON object or a structured fallback error record.
    """
    # --- Strategy 1: Direct parse ---
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # --- Strategy 2: Strip markdown fences ---
    stripped = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
    stripped = stripped.replace("```", "").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # --- Strategy 3: String-slice brace extraction ---
    brace_open = raw_text.find("{")
    brace_close = raw_text.rfind("}")
    if brace_open != -1 and brace_close != -1 and brace_close > brace_open:
        candidate = raw_text[brace_open : brace_close + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # --- Strategy 4: Graceful failure record ---
    return {
        "parse_error": True,
        "raw_preview": raw_text[:300],
        "message": "All JSON recovery strategies exhausted. Raw preview captured.",
    }


# ==============================================================================
# SECTION 4: API CALL LAYER  [v1.1 PATCHED]
# Now returns a (success: bool, content: str) tuple instead of a raw string.
# This completely separates network/auth failures from JSON parse failures,
# so the real error reason is always surfaced in the console.
# ==============================================================================


def call_groq_api(
    system_prompt: str,
    user_message: str,
    agent_label: str
) -> tuple:
    """
    Production-grade Groq API client.

    Features:
      - Adaptive retry handling for HTTP 429
      - Exponential backoff
      - Connection recovery
      - Timeout recovery
      - Explicit error propagation

    Returns:
        (True, response_content)
        (False, error_message)
    """

    print(
        f"    [NET] {agent_label} — "
        f"Enforcing {RATE_LIMIT_SLEEP_SECONDS}s rate-limit guard..."
    )

    time.sleep(RATE_LIMIT_SLEEP_SECONDS)

    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "temperature": 0.1,
        "max_tokens": 350,
    }

    MAX_RETRIES = 6

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.post(
                GROQ_ENDPOINT,
                headers=build_headers(),
                json=payload,
                timeout=30,
            )

            # ------------------------------------------------------
            # SUCCESS
            # ------------------------------------------------------
            if response.ok:

                data = response.json()

                content = (
                    data["choices"][0]
                    ["message"]
                    ["content"]
                )

                return True, content

            # ------------------------------------------------------
            # HTTP 429 RATE LIMIT
            # ------------------------------------------------------
            if response.status_code == 429:

                wait_time = min(
                    60,
                    2 ** attempt
                )

                try:
                    body = response.json()

                    message = body["error"]["message"]

                    retry_match = re.search(
                        r"try again in ([0-9.]+)s",
                        message
                    )

                    if retry_match:

                        wait_time = (
                            float(
                                retry_match.group(1)
                            ) + 1
                        )

                except Exception:
                    pass

                print(
                    f"    [429 RATE LIMIT] "
                    f"Attempt {attempt}/{MAX_RETRIES}"
                )

                print(
                    f"    [BACKOFF] "
                    f"Sleeping {wait_time:.1f}s before retry..."
                )

                time.sleep(wait_time)

                continue

            # ------------------------------------------------------
            # OTHER HTTP ERRORS
            # ------------------------------------------------------
            try:
                error_body = response.json()
            except Exception:
                error_body = response.text[:400]

            return (
                False,
                (
                    f"HTTP {response.status_code} "
                    f"from Groq API. "
                    f"Body: {error_body}"
                ),
            )

        except requests.exceptions.Timeout:

            backoff = min(
                60,
                2 ** attempt
            )

            print(
                f"    [TIMEOUT] "
                f"Attempt {attempt}/{MAX_RETRIES}"
            )

            print(
                f"    [BACKOFF] "
                f"Retrying in {backoff}s..."
            )

            time.sleep(backoff)

        except requests.exceptions.ConnectionError as conn_err:

            backoff = min(
                60,
                2 ** attempt
            )

            print(
                f"    [CONNECTION ERROR]"
            )

            print(
                f"    Detail: {conn_err}"
            )

            print(
                f"    Retrying in {backoff}s..."
            )

            time.sleep(backoff)

        except (KeyError, IndexError) as parse_err:

            return (
                False,
                (
                    "RESPONSE_STRUCTURE_ERROR — "
                    f"{parse_err}"
                ),
            )

    return (
        False,
        (
            "RETRY_EXHAUSTED — "
            "Maximum retry attempts reached."
        ),
    )

# ==============================================================================
# SECTION 5: AGENT SYSTEM PROMPTS
# ==============================================================================

# ------------------------------------------------------------------------------
# AGENT 1 SYSTEM PROMPT — Strategic Context Triage
# Evaluates raw domain inputs, scores them, extracts metadata, validates structure.
# ------------------------------------------------------------------------------

AGENT_1_SYSTEM_PROMPT = """
You are the Strategic Context Triage Agent operating within an enterprise
multi-agent decision system. Your function is to evaluate raw domain data
submissions and produce a structured operational assessment.

For every input you receive, respond ONLY with a valid JSON object using
this exact schema (no markdown, no commentary, no backticks):

{
  "record_id": "<the record_id from the input>",
  "domain": "<domain name>",
  "impact_score": <integer 1-10>,
  "score_rationale": "<one concise sentence explaining the score>",
  "primary_risk_flags": ["<flag1>", "<flag2>", "<flag3>"],
  "structural_validity": "VALID",
  "validity_notes": "<brief note on data completeness or anomalies>",
  "domain_metadata": {
    "urgency": "CRITICAL",
    "stakeholder_tier": "<who is immediately affected>",
    "data_freshness": "REAL-TIME"
  },
  "triage_summary": "<two sentences max summarizing the situation>"
}

URGENCY options: CRITICAL, HIGH, MEDIUM, LOW
DATA_FRESHNESS options: REAL-TIME, RECENT, HISTORICAL

SCORING RUBRIC:
  9-10: Imminent life-safety risk, systemic institutional failure, or
        irreversible harm within 24-72 hrs.
  7-8:  High operational urgency, measurable degradation, escalation required.
  5-6:  Moderate concern, monitoring needed, no immediate crisis.
  3-4:  Low-level friction, addressable through routine channels.
  1-2:  Informational only, no actionable risk detected.

Be precise. Be objective. Output ONLY the JSON object. No extra text.
""".strip()

# ------------------------------------------------------------------------------
# AGENT 2 SYSTEM PROMPT — Executive Content Engine
# Intercepts high-impact records, authors executive briefs, appends action links.
# ------------------------------------------------------------------------------

AGENT_2_SYSTEM_PROMPT = """
You are the Executive Content Engine, the second stage of an enterprise
agentic pipeline. You receive triage records scored >= 7 by the upstream
Strategic Context Triage Agent. Author a polished executive response brief
and recommend one concrete operational next step.

Respond ONLY with a valid JSON object using this exact schema
(no markdown, no commentary, no backticks):

{
  "record_id": "<same record_id from triage input>",
  "domain": "<same domain>",
  "executive_brief": "<3-4 sentence authoritative summary written for a C-suite or senior official audience. Be specific, cite key figures from the original data, and communicate urgency proportionate to the score.>",
  "recommended_action": "<one clear, specific, actionable directive — not generic advice>",
  "action_link": "<a plausible real-format URL to a relevant authoritative resource, framework, protocol, or dashboard. Format: https://domain.org/path>",
  "escalation_tier": "C-SUITE",
  "response_deadline": "<human-readable deadline e.g. Within 6 hours>",
  "agent_signature": "Executive Content Engine v1.1 — Groq/llama-3.1-8b-instant"
}

ESCALATION_TIER options: BOARD, C-SUITE, DEPARTMENT-HEAD, OPERATIONS

Write with authority.

STRICT RULES:

- Never invent budgets, funding amounts, organizations, statistics, policies, committees, or agencies.
- Use only facts present in the original source data.
- Do not fabricate URLs.
- If no authoritative URL is known, set:

"action_link": "N/A"

- Recommended actions must be operational directives only.
- Do not propose dollar amounts.
- Maintain executive tone suitable for senior leadership.

Output ONLY the JSON object.
No markdown.
No commentary.
No extra text.

 IMPORTANT:

Never invent budgets, funding amounts,
organizations, statistics, or URLs.

Only recommend actions directly supported
by the source data.

If a numerical value is absent from the
source, do not create one.

If no authoritative URL is known,
set:

"action_link": "N/A" """.strip()


# ==============================================================================
# SECTION 6: AGENT EXECUTION FUNCTIONS  [v1.1 PATCHED]
# Both agents now unpack the (bool, str) tuple from call_groq_api() and
# separately handle API failures vs JSON parse failures.
# ==============================================================================


def run_agent_1_triage(record: dict) -> dict:
    """
    Agent 1: Strategic Context Triage.

    Sends a domain input record to the LLM for structured scoring,
    metadata extraction, and validity checking.

    PATCH v1.1:
      - Unpacks (success, content) tuple from call_groq_api().
      - Prints the real API error if success is False.
      - Only attempts JSON parsing when the API call actually succeeded.

    Args:
        record (dict): A single entry from INPUT_MATRIX.

    Returns:
        dict: Parsed triage assessment, or structured error record.
    """
    user_message = (
        f"RECORD_ID: {record['record_id']}\n"
        f"DOMAIN: {record['domain']}\n"
        f"PAYLOAD:\n{record['payload']}"
    )

    success, content = call_groq_api(
        system_prompt=AGENT_1_SYSTEM_PROMPT,
        user_message=user_message,
        agent_label="Agent-1/StrategicTriage",
    )

    # --- Network / auth / HTTP failure ---
    if not success:
        print(f"  [API ERROR] {content}")
        return {
            "parse_error": True,
            "error_type": "API_CALL_FAILED",
            "error_detail": content,
            "message": (
                "Agent 1 API call failed — see error_detail above for root cause. "
                "Common fixes: (1) Replace ACTIVE_KEY with your real Groq key. "
                "(2) Check internet. (3) Increase RATE_LIMIT_SLEEP_SECONDS if HTTP 429."
            ),
        }

    # --- API succeeded — now parse the JSON ---
    parsed = safe_json_parse(content)

    if "parse_error" in parsed:
        print(f"  [JSON ERROR] LLM returned non-JSON content.")
        print(f"  [RAW PREVIEW] {content[:250]}")
        parsed["error_type"] = "JSON_PARSE_FAILED"
        parsed["_agent"] = "Agent-1/StrategicTriage"
        return parsed

    parsed["_agent"] = "Agent-1/StrategicTriage"
    parsed["_raw_response_length"] = len(content)
    return parsed


def run_agent_2_executive(triage_result: dict, original_payload: str) -> dict:
    """
    Agent 2: Executive Content Engine.

    Consumes a high-impact triage result and produces a polished executive
    brief with a concrete action link.

    PATCH v1.1:
      - Unpacks (success, content) tuple from call_groq_api().
      - Prints the real API error if success is False.
      - Only attempts JSON parsing when the API call actually succeeded.

    Args:
        triage_result    (dict): The structured output from Agent 1.
        original_payload (str) : Original domain data string for full context.

    Returns:
        dict: Parsed executive brief, or structured error record.
    """
    user_message = (
        f"TRIAGE ASSESSMENT:\n{json.dumps(triage_result, indent=2)}\n\n"
        f"ORIGINAL SOURCE DATA:\n{original_payload}"
    )

    success, content = call_groq_api(
        system_prompt=AGENT_2_SYSTEM_PROMPT,
        user_message=user_message,
        agent_label="Agent-2/ExecutiveEngine",
    )

    # --- Network / auth / HTTP failure ---
    if not success:
        print(f"  [API ERROR] {content}")
        return {
            "parse_error": True,
            "error_type": "API_CALL_FAILED",
            "error_detail": content,
        }

    # --- API succeeded — now parse the JSON ---
    parsed = safe_json_parse(content)

    if "parse_error" in parsed:
        print(f"  [JSON ERROR] Agent 2 returned non-JSON content.")
        print(f"  [RAW PREVIEW] {content[:250]}")
        parsed["error_type"] = "JSON_PARSE_FAILED"

    parsed["_agent"] = "Agent-2/ExecutiveEngine"
    parsed["_raw_response_length"] = len(content)
    return parsed


# ==============================================================================
# SECTION 7: MAIN ORCHESTRATION LOOP
# ==============================================================================


def orchestrate() -> None:
    """
    Primary orchestration function — the Multi-Agent Loop.

    Execution Flow:
      1. Initialize global state container.
      2. For each domain record in INPUT_MATRIX:
         a. Route through Agent 1 (Strategic Context Triage).
         b. Evaluate impact_score against HIGH_IMPACT_THRESHOLD.
         c. If score >= threshold: route through Agent 2 (Executive Engine).
         d. Accumulate all outputs into global_state.
      3. Generate a comprehensive status summary.
      4. Commit final memory state to STATE_OUTPUT_FILE (global_state.json).
    """

    run_timestamp = datetime.now(timezone.utc).isoformat()

    print("=" * 72)
    print("  ENTERPRISE MULTI-AGENT LOOP SYSTEM v1.1 — INITIALIZING")
    print(f"  Run Timestamp : {run_timestamp}")
    print(f"  Model         : {MODEL_ID}")
    print(f"  Endpoint      : {GROQ_ENDPOINT}")
    print(f"  Input Records : {len(INPUT_MATRIX)}")
    print(f"  Impact Gate   : Score >= {HIGH_IMPACT_THRESHOLD}")
    print("=" * 72)

    # --------------------------------------------------------------------------
    # Global state container — persisted to disk at end of run
    # --------------------------------------------------------------------------
    global_state = {
        "system": "Enterprise Multi-Agent Loop v1.1",
        "run_timestamp": run_timestamp,
        "model": MODEL_ID,
        "endpoint": GROQ_ENDPOINT,
        "config": {
            "high_impact_threshold": HIGH_IMPACT_THRESHOLD,
            "rate_limit_sleep_seconds": RATE_LIMIT_SLEEP_SECONDS,
            "total_input_records": len(INPUT_MATRIX),
        },
        "records": [],   # Per-record detailed state
        "summary": {},   # Populated after loop completion
    }

    # Tracking counters for summary block
    total_processed    = 0
    total_high_impact  = 0
    total_agent2_ok    = 0
    total_errors       = 0
    domain_scores      = {}

    # --------------------------------------------------------------------------
    # MAIN LOOP — iterate over each domain record
    # --------------------------------------------------------------------------
    for idx, record in enumerate(INPUT_MATRIX, start=1):

        record_id = record["record_id"]
        domain    = record["domain"]

        print(f"\n{'─' * 72}")
        print(f"  [{idx}/{len(INPUT_MATRIX)}] Processing: {record_id} | Domain: {domain}")
        print(f"{'─' * 72}")

        # Accumulator for this record's state
        record_state = {
            "record_id":         record_id,
            "domain":            domain,
            "processing_order":  idx,
            "agent_1_triage":    None,
            "agent_2_executive": None,
            "routing_decision":  None,
            "error":             None,
        }

        # ------------------------------------------------------------------
        # STAGE A: Agent 1 — Strategic Context Triage
        # ------------------------------------------------------------------
        print(f"\n  >> STAGE A — Agent 1: Strategic Context Triage")

        triage_result = run_agent_1_triage(record)

        # Detect any kind of failure from Agent 1
        is_agent1_error = (
            "parse_error" in triage_result
            or "_agent" not in triage_result
        )

        if is_agent1_error:
            error_type   = triage_result.get("error_type", "UNKNOWN")
            error_detail = triage_result.get("error_detail", "")
            print(f"  [SKIP] {record_id} — Agent 1 failed ({error_type}).")
            if error_detail:
                print(f"  [DETAIL] {error_detail}")
            record_state["error"]            = triage_result
            record_state["routing_decision"] = "ERROR_SKIP"
            total_errors += 1
            global_state["records"].append(record_state)
            total_processed += 1
            continue

        record_state["agent_1_triage"] = triage_result

        # Extract scoring fields safely
        impact_score    = triage_result.get("impact_score", 0)
        score_rationale = triage_result.get("score_rationale", "N/A")
        urgency         = triage_result.get("domain_metadata", {}).get("urgency", "UNKNOWN")

        domain_scores[record_id] = {
            "domain":       domain,
            "impact_score": impact_score,
            "urgency":      urgency,
        }

        print(f"  [TRIAGE] Impact Score : {impact_score}/10")
        print(f"  [TRIAGE] Urgency      : {urgency}")
        print(f"  [TRIAGE] Rationale    : {score_rationale}")

        total_processed += 1

        # ------------------------------------------------------------------
        # ROUTING GATE: Score >= HIGH_IMPACT_THRESHOLD?
        # ------------------------------------------------------------------
        if impact_score >= HIGH_IMPACT_THRESHOLD:
            total_high_impact += 1
            record_state["routing_decision"] = "ESCALATED_TO_AGENT_2"
            print(
                f"\n  [GATE] Score {impact_score} >= {HIGH_IMPACT_THRESHOLD}. "
                f"Routing to Agent 2."
            )

            # --------------------------------------------------------------
            # STAGE B: Agent 2 — Executive Content Engine
            # --------------------------------------------------------------
            print(f"\n  >> STAGE B — Agent 2: Executive Content Engine")

            executive_result = run_agent_2_executive(
                triage_result=triage_result,
                original_payload=record["payload"],
            )

            if "parse_error" in executive_result:
                error_type   = executive_result.get("error_type", "UNKNOWN")
                error_detail = executive_result.get("error_detail", "")
                print(f"  [WARN] Agent 2 failed for {record_id} ({error_type}).")
                if error_detail:
                    print(f"  [DETAIL] {error_detail}")
                record_state["error"] = executive_result
            else:
                total_agent2_ok += 1
                record_state["agent_2_executive"] = executive_result

                # Surface key outputs to console
                brief    = executive_result.get("executive_brief",    "N/A")
                action   = executive_result.get("recommended_action", "N/A")
                link     = executive_result.get("action_link",        "N/A")
                tier     = executive_result.get("escalation_tier",    "N/A")
                deadline = executive_result.get("response_deadline",  "N/A")

                print(f"\n  [EXEC BRIEF]\n  {brief}")
                print(f"\n  [ACTION]     {action}")
                print(f"  [LINK]       {link}")
                print(f"  [TIER]       {tier} | DEADLINE: {deadline}")

        else:
            # Below threshold — log and skip Agent 2
            record_state["routing_decision"] = "BELOW_THRESHOLD_LOGGED"
            print(
                f"\n  [GATE] Score {impact_score} < {HIGH_IMPACT_THRESHOLD}. "
                f"Record logged. Agent 2 not invoked."
            )
            triage_summary = triage_result.get("triage_summary", "N/A")
            print(f"  [SUMMARY] {triage_summary}")

        global_state["records"].append(record_state)

    # --------------------------------------------------------------------------
    # SECTION 8: STATUS SUMMARY GENERATION
    # --------------------------------------------------------------------------

    print(f"\n{'=' * 72}")
    print("  ORCHESTRATION COMPLETE — STATUS SUMMARY")
    print(f"{'=' * 72}")

    below_threshold = total_processed - total_high_impact - total_errors

    summary = {
        "run_timestamp":            run_timestamp,
        "total_records_processed":  total_processed,
        "high_impact_records":      total_high_impact,
        "below_threshold_records":  below_threshold,
        "agent_2_executions":       total_agent2_ok,
        "errors_encountered":       total_errors,
        "agent_2_success_rate": (
            f"{round((total_agent2_ok / total_high_impact) * 100, 1)}%"
            if total_high_impact > 0
            else "N/A"
        ),
        "domain_score_index": domain_scores,
        "highest_impact_record": (
            max(
                domain_scores,
                key=lambda k: domain_scores[k]["impact_score"],
            )
            if domain_scores
            else "None"
        ),
    }

    global_state["summary"] = summary

    # Human-readable summary block
    print(f"  Total Records Processed   : {summary['total_records_processed']}")
    print(f"  High-Impact Escalations   : {summary['high_impact_records']}")
    print(f"  Below-Threshold (Logged)  : {summary['below_threshold_records']}")
    print(f"  Agent 2 Executions        : {summary['agent_2_executions']}")
    print(f"  Agent 2 Success Rate      : {summary['agent_2_success_rate']}")
    print(f"  Errors Encountered        : {summary['errors_encountered']}")
    print(f"  Highest Impact Record     : {summary['highest_impact_record']}")

    if domain_scores:
        print(f"\n  Domain Score Index:")
        for rid, meta in domain_scores.items():
            filled = "#" * meta["impact_score"]
            empty  = "-" * (10 - meta["impact_score"])
            print(
                f"    {rid} | {meta['domain']:<28} "
                f"| [{filled}{empty}] {meta['impact_score']:>2}/10 "
                f"| {meta['urgency']}"
            )
    else:
        print("\n  Domain Score Index: (no records scored — check API key and connection)")

    # --------------------------------------------------------------------------
    # SECTION 9: PERSIST MEMORY STATE TO DISK
    # --------------------------------------------------------------------------
    print(f"\n{'─' * 72}")
    print(f"  Committing global memory state to: {STATE_OUTPUT_FILE}")

    try:
        with open(STATE_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(global_state, f, indent=2, ensure_ascii=False)
        file_size = len(json.dumps(global_state))
        print(f"  [OK] State committed successfully.")
        print(f"       File: {STATE_OUTPUT_FILE}")
        print(f"       Size: {file_size} bytes")
    except IOError as io_err:
        print(f"  [ERROR] Failed to write state file: {io_err}")

    print(f"\n{'=' * 72}")
    print("  MULTI-AGENT LOOP TERMINATED CLEANLY")
    print(f"{'=' * 72}\n")


# ==============================================================================
# SECTION 10: ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    orchestrate()

