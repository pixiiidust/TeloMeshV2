🧱 TeloMesh — SIMPLE Stage (MVP Implementation Flow)
🎯 Goal:
Simulate event data → parse into sessions → build a directed, event-typed graph of user flows.

# Refer to /docs/cursor_rules.md for detailed TDD and behavior contracts


1️⃣ 🔧 Synthetic Event Generator (synthetic_event_generator.py)
    📂 Location:
    data/synthetic_event_generator.py

    ✅ Purpose:
    Creates events.csv that mimics Amplitude/Mixpanel exports.

    📊 Output Columns:
    Column	Type	Example
    user_id	string	user_001
    timestamp	ISO8601	2025-06-05T12:30:00Z
    page_url	string	/search
    event_name	string	Clicked Search

    👤 100–10000 users
    📈 500 total events per 100 users
    🔁 3–7 events per session
    🕒 30 sec–5 min spacing; 30 min idle gap triggers session break

2️⃣ 📥 Session Parser (parse_sessions.py)
    📂 Location:
    ingest/parse_sessions.py

    ✅ Purpose:
    Reads data/events.csv and writes outputs/session_flows.csv.

    🔧 Key Steps:
    1. Read and sort events by user_id + timestamp
    2. Split into sessions using 30 min idle timeout
    3. Format each session as:
    python
    [
        {"page": "/home", "event": "Viewed Page", "timestamp": ...},
        ...
    ]
    4. Save to outputs/session_flows.csv as sequences:
    user_id,session_id,page,event,timestamp
    user_001,session_001,/home,Viewed Page,2025-06-05T12:00:00Z
    ...

3️⃣ 🌉 Flow Graph Builder (build_graph.py)
    📂 Location:
    ingest/build_graph.py

    ✅ Purpose:
    Builds a NetworkX directed graph from session_flows.csv.

    🔧 Logic:
    For each session:
    1. For each event_n → event_n+1, define:
        - from_page = event_n['page']
        - to_page = event_n+1['page']
        - edge_event = event_n+1['event']
    2. Create or update edge with:
        python
        G.add_edge(from_page, to_page, event=edge_event, weight=1)
    3. If same (from, to, event) exists, increment weight

    💾 Output:
    outputs/user_graph.gpickle (serialized NetworkX graph)


QA Module — Flow Metrics (T4.x)
File: ingest/flow_metrics.py
Purpose: Perform lightweight checks on session richness and graph integrity post-construction.

Validations:

sessions_per_user ≥ 1

average flow length ≥ 3

most common event exists

session count ≥ 50

node count ≥ 10

Outputs:
logs/session_stats.log (text summary)

Tests (T4.x):

python
Copy
Edit
def test_stats_log_created():
    assert os.path.exists("logs/session_stats.log")

def test_sessions_per_user_valid():
    # Load session_flows and validate ≥ 1 per user

📁 SIMPLE File Structure

TeloMesh/
    ├── data/
    │   ├── events.csv                   # Synthetic event log
    │   └── synthetic_event_generator.py # (Optional: run to create test data)
    ├── ingest/
    │   ├── parse_sessions.py            # CSV → sessions
    │   └── build_graph.py               # sessions → graph
    ├── outputs/
    │   ├── session_flows.csv            # Flat session log
    │   └── user_graph.gpickle           # Directed, typed, weighted graph

✅ SIMPLE Stage Success Criteria
    Component	Output File	Check
    Synthetic Events	data/events.csv	500+ rows, realistic flows
    Session Parser	outputs/session_flows.csv	Each session = clean ordered path
    Graph Builder	outputs/user_graph.gpickle	Directed graph with typed, weighted edges

🧠 Edge Definition (MVP Standard)
    Each edge is:
    - Directed: from_page → to_page
    - Triggered by: single event type
    - Weighted: number of occurrences

Edge metadata:
    python
    {
        "event": "Clicked Product",
        "weight": 5
    }

🧱 TeloMesh — SIMPLE Stage (MVP) Test-Driven Loop Based on #cursorRule

    ✅ Global Rule — Event Volume Scaling
        For any generated dataset:
        num_events ≥ 5 × num_users
        This is a hard constraint for synthetic data generation. Enforced via:
        python
        assert len(df) >= 5 * df["user_id"].nunique()
    ✅ Cursor Rule (Global)
        markdown
        #cursorRule

        Always follow a Test-Driven Development (TDD) loop in small, testable increments:
        1. Define a small, well-scoped feature or task.
        2. Write a failing test that describes the expected behavior.
        3. Generate only enough code to pass the test.
        4. Run the test.
        5. If the test fails, analyze and revise.
        6. Once it passes, commit and continue.

        Never write full features before writing tests.

🔧 1️⃣ File: data/synthetic_event_generator.py
    🎯 Goal:
    Generate realistic synthetic events.csv for 100–10,000 users with at least 5 × users events.

    🧪 Tests (T1.x) — MUST BE WRITTEN FIRST
    python
    def test_events_csv_exists():
        assert os.path.exists("data/events.csv")

    def test_minimum_rows_per_user():
        df = pd.read_csv("data/events.csv")
        num_users = df["user_id"].nunique()
        assert len(df) >= 5 * num_users, f"Expected at least {5*num_users} events"

    def test_required_columns_exist():
        df = pd.read_csv("data/events.csv")
        required = {"user_id", "timestamp", "page_url", "event_name"}
        assert required.issubset(df.columns)

    def test_timestamps_parse():
        df = pd.read_csv("data/events.csv")
        pd.to_datetime(df["timestamp"], errors="raise")  # Will raise if malformed

    def test_minimum_user_count():
        df = pd.read_csv("data/events.csv")
        assert df["user_id"].nunique() >= 30
    🔁 Cursor Execution Loop
    python
    while not all_tests_T1_passed:
        # Do NOT write event generator code until tests exist
        write only enough code to pass next failing T1 test
        run tests
        fix failures
    log("✅ synthetic_event_generator.py complete — all T1 tests pass")

📥 2️⃣ File: ingest/parse_sessions.py
    🎯 Goal:
    Convert events.csv into clean, chronologically ordered session_flows.csv using idle gap logic.

    🧪 Tests (T2.x) — MUST BE WRITTEN FIRST
    python
    def test_session_csv_created():
        assert os.path.exists("outputs/session_flows.csv")

    def test_session_columns_exist():
        df = pd.read_csv("outputs/session_flows.csv")
        required = {"user_id", "session_id", "page", "event", "timestamp"}
        assert required.issubset(df.columns)

    def test_minimum_sessions_exist():
        df = pd.read_csv("outputs/session_flows.csv")
        assert df["session_id"].nunique() >= 100

    def test_sessions_chronological():
        df = pd.read_csv("outputs/session_flows.csv")
        grouped = df.groupby("session_id")["timestamp"]
        assert all(pd.to_datetime(grouped.get_group(sid)).is_monotonic_increasing for sid in grouped.groups)

    def test_idle_gaps_handled():
        df = pd.read_csv("outputs/session_flows.csv")
        # Optional: validate known synthetic idle break behavior or max intra-session delta < 30min
    🔁 Cursor Execution Loop
    python
    while not all_tests_T2_passed:
        write minimal parse_sessions.py to satisfy next failing T2 test
        run tests
        fix failures
    log("✅ parse_sessions.py complete — all T2 tests pass")

🌉 3️⃣ File: ingest/build_graph.py
    🎯 Goal:
    Build a directed, event-typed graph (user_graph.gpickle) from parsed sessions.

    🧪 Tests (T3.x) — MUST BE WRITTEN FIRST
    python
    def test_graph_file_created():
        assert os.path.exists("outputs/user_graph.gpickle")

    def test_graph_node_and_edge_count():
        G = nx.read_gpickle("outputs/user_graph.gpickle")
        assert len(G.nodes) >= 10
        assert len(G.edges) >= 15

    def test_edge_attributes_exist():
        G = nx.read_gpickle("outputs/user_graph.gpickle")
        for _, _, data in G.edges(data=True):
            assert "event" in data and "weight" in data

    def test_edge_weights_valid():
        G = nx.read_gpickle("outputs/user_graph.gpickle")
        assert all(data["weight"] >= 1 for _, _, data in G.edges(data=True))

    def test_unique_event_per_edge():
        G = nx.read_gpickle("outputs/user_graph.gpickle")
        seen = {}
        for u, v, d in G.edges(data=True):
            key = (u, v)
            assert key not in seen or seen[key] == d["event"]
            seen[key] = d["event"]

    🔁 Cursor Execution Loop
    while not all_tests_T3_passed:
        write minimal build_graph.py to pass next failing test
        run tests
        fix failures
    log("✅ build_graph.py complete — all T3 tests pass")

📁 SIMPLE Stage Directory Structure
    TeloMesh/
    ├── data/
    │   ├── events.csv                   # ← T1 output
    │   └── synthetic_event_generator.py
    ├── ingest/
    │   ├── parse_sessions.py            # ← T2 output
    │   └── build_graph.py               # ← T3 output
    ├── outputs/
    │   ├── session_flows.csv
    │   └── user_graph.gpickle
    ├── tests/
    │   └── test_simple.py               # all test cases for T1–T3

✅ Summary Table
    File	Test Prefix	Key Rule Enforced
    synthetic_event_generator.py	T1.x	events ≥ 5 × users
    parse_sessions.py	T2.x	Idle split + chronological sessions
    build_graph.py	T3.x	Directed edges, typed, weighted, unique


🧱 SIMPLE Stage – Extra Cursor Guidance 
File: parse_sessions.py + build_graph.py + flow_metrics.py

📌 1. Session Flow Construction (parse_sessions.py)
python
Copy
Edit
# CURSOR RULE: SESSION FLOW PARSING
# - Read raw_events.csv with columns: session_id, user_id, timestamp, page, event
# - Sort chronologically by session_id, then timestamp
# - For each session, emit ordered triplets: (from_page, to_page, event)
# - Output: outputs/session_flows.csv must contain at least 500 rows per 100 users
# - Output must include: session_id, from_page, to_page, event, timestamp
📌 2. User Graph Building (build_graph.py)
python
Copy
Edit
# CURSOR RULE: USER FLOW GRAPH CONSTRUCTION
# - Nodes = unique pages
# - Directed edges = (from_page → to_page)
# - Edge attributes must include: 'event', 'weight' = count of traversals
# - Output: outputs/user_graph.gpickle — a NetworkX DiGraph
# - Validate: ≥1 connected component and ≥2 unique edge events
📌 3. Session + Graph QA (flow_metrics.py)
python
Copy
Edit
# CURSOR RULE: BASIC SESSION + GRAPH VALIDATION
# - Compute:
#     - sessions_per_user ≥ 1
#     - average flow length ≥ 3
#     - most common event type (e.g., "Clicked CTA")
# - Output test logs to logs/session_stats.log
# - Raise warning if total sessions < 50 or nodes < 10
