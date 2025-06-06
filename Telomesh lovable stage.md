💛 TeloMesh — LOVABLE Stage (MVP Implementation Flow)
🎯 Goal
Identify and rank the most fragile user flows using page-event pairs that score high on both exit likelihood and structural centrality. Compute a WSJF_Friction_Score per node and per flow, then visualize this interactively for PMs with heatmaps, tooltips, and session-based flow summaries.

# Refer to /docs/cursor_rules.md for detailed TDD and behavior contracts


1️⃣ 🔍 Event + Flow Friction Analyzer — analysis/event_chokepoints.py
    📂 Location
    analysis/event_chokepoints.py

    ✅ Purpose
    Detect UX weak points by analyzing:
    Event-level exit rates: Probability session ends after (page, event)
    Page betweenness: How structurally critical a page is in session paths
    Fragile flows: Flows where ≥2 high-friction (page, event) steps occur

    🔢 Core Formula

    python
    Copy
    Edit
    WSJF_Friction_Score = exit_rate × betweenness
    📥 Inputs

    outputs/session_flows.csv ← parsed from raw session events

    outputs/user_graph.gpickle ← graph built with weighted (page, event) edges

    📤 Outputs

    File	Description
    event_chokepoints.csv	Flat table with (page, event) friction metrics
    high_friction_flows.csv	Session-level flows with ≥2 chokepoints
    friction_node_map.json	Node + edge WSJF scores for rendering

    📊 Output Schema (event_chokepoints.csv)

    Column	Type	Description
    page	string	e.g., /checkout
    event	string	e.g., Clicked Confirm
    exit_rate	float	Share of sessions ending here
    betweenness	float	Page’s centrality in all flows
    users_lost	int	Number of users exiting here
    WSJF_Friction_Score	float	Prioritization score

2️⃣ 📊 Friction Intelligence Dashboard — ui/dashboard.py
    📂 Location
    ui/dashboard.py

    ✅ Purpose
    Give PMs an intuitive, interactive lens to explore UX friction:

    Top Friction Insights
        Toggle between Top 10% / 20% nodes by WSJF
        Sort + filter friction table by score or event type

    Flow Breakdown
        Show flows like:
        /search (Typed Query) → /product (Clicked Add to Cart) → /checkout (Clicked Confirm)
        Highlight if multiple nodes in the flow rank high
    
    Graph Heatmap
        Nodes colored by percentile (🔴 = Top 10%)
        Edges show event names
        Clicking a node opens:
            Event breakdown at that node
            Affected sessions and flows
    
    Tooltips Everywhere
        betweenness: "How many flows pass through this page?"
        exit_rate: "How often do users exit after this event?"
        WSJF: "Composite urgency = exit × criticality"

    📥 Dependencies

    event_chokepoints.csv
    high_friction_flows.csv
    user_graph.gpickle
    friction_node_map.json

    🎨 Key UI Elements

    Component	Description
    Friction Table	Sortable table by WSJF_Friction_Score, filterable by event/page
    Flow Summary Viewer	Shows fragile flows with ≥2 high-friction (page, event) pairs
    Graph Heatmap	Visual graph with WSJF-colored nodes, edge-labels show events
    Click-to-Drilldown	Node click opens session breakdown + top events at that node
    Tooltips	Render metric definitions in PM-friendly terms

📁 LOVABLE Stage Directory Structure

    TeloMesh/
    ├── analysis/
    │   └── event_chokepoints.py
    ├── ui/
    │   └── dashboard.py
    ├── outputs/
    │   ├── event_chokepoints.csv
    │   ├── high_friction_flows.csv
    │   └── friction_node_map.json

✅ LOVABLE Stage Success Criteria
    Component	Output File	Success Check
    Event Analyzer	event_chokepoints.csv	≥10 rows, valid score ranges
    Fragile Flow Detector	high_friction_flows.csv	≥2 flows with ≥2 chokepoints
    Dashboard	Streamlit UI	Loads table, graph, tooltips, drilldowns
    PM Utility	—	1+ actionable flow detected, all tooltips render

🧠 Friction Metrics
    🧩 Node-Level
    Each (page, event) is scored by:
    exit_rate: Probability of session ending after the event
    betweenness: Page’s role in flow structure
    WSJF_Friction_Score: Exit × structure

    🔁 Flow-Level
    A fragile flow = session with ≥2 high-scoring events
    Helps reveal "cascading UX failure paths" vs isolated issues

💛 Test-Driven Loop — LOVABLE Stage (#cursorRule)

🔧 analysis/event_chokepoints.py
🧪 Tests (L1.x)

    python
    Copy
    Edit
    def test_event_chokepoint_csv_exists(): ...
    def test_required_columns(): ...
    def test_valid_score_ranges(): ...
    def test_fragile_flows_file_exists(): ...
    def test_output_node_map(): ...
    🔁 Cursor Loop

    python
    Copy
    Edit
    while not all_L1_tests_pass:
        write just enough logic to pass next test
        run tests
        repeat until ✅
        
📊 ui/dashboard.py
    🧪 Tests (L2.x)

    python
    Copy
    Edit
    def test_dashboard_loads(): ...
    def test_tooltip_exists(): ...
    def test_heatmap_nodes_colored(): ...
    def test_node_click_shows_events(): ...
    🔁 Cursor Loop

    python
    Copy
    Edit
    while not all_L2_tests_pass:
        increment UI logic (table → graph → drilldowns)
        run tests
        repeat until ✅


💛 LOVABLE Stage – Extra Cursor Guidance
File: event_chokepoints.py + dashboard.py

📌 1. Event Chokepoint Analysis (event_chokepoints.py)
python
Copy
Edit
# CURSOR RULE: CHOKEPOINT METRIC COMPUTATION
# - Input: session_flows.csv + user_graph.gpickle
# - Output: event_chokepoints.csv with columns:
#     ['page', 'event', 'exit_rate', 'betweenness', 'users_lost', 'WSJF_Friction_Score']
# - WSJF_Friction_Score = exit_rate × betweenness
# - Identify top 10% WSJF scores to define “chokepoints”
# - Output: high_friction_flows.csv → sessions containing ≥2 chokepoints
# - Output: friction_node_map.json → {page: WSJF score of highest event at that page}
# - Ensure that session order is preserved when detecting fragile flows

📌 2. Friction Intelligence Dashboard (dashboard.py)
python
Copy
Edit
# CURSOR RULE: PM-FACING DASHBOARD CONSTRUCTION
## 🎯 Purpose:
Build a PM-intuitive, **dark-themed**, Streamlit dashboard that reveals session friction points without requiring deep analytics knowledge. Prioritize clarity, minimalism, and interactivity — using only pre-computed files.

## 🔍 Inputs Required:
- `event_chokepoints.csv` → ranked (page, event) pairs with friction scores
- `high_friction_flows.csv` → session paths with ≥2 chokepoints
- `friction_node_map.json` → WSJF node scores for heatmap
- `user_graph.gpickle` → user flow graph with edges containing `event` labels

## 📊 Required Components:

| Component             | Behavior                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| 🟫 Friction Table      | - Sortable by `WSJF_Friction_Score`<br>- Filterable by `page`, `event`      |
| 🔁 Flow Summary        | - Displays user paths with 2+ chokepoints (e.g. `/search → /product → /cart`)<br>- Highlights cumulative friction score across flow |
| 🌐 Graph Heatmap       | - - **Dark-themed node-link graph**<br>- Top 10% (🔴) and 20% (🟠) friction nodes<br>- Edge hover / clicking a node reveals event label |
| 🖱️ Click-to-Drilldown  | - Shows:<br>• Event breakdown<br>• Users lost<br>• Incoming/outgoing links |
| 💬 Metric Tooltips     | - Betweenness: “How central this page is in user flows”<br>- Exit rate: “% of users who left at this step”<br>- WSJF: “Frustration × importance” |

## 🔍 Interactivity & UX Guidance:

- **Keep graphs uncluttered**: Only show top friction nodes. Limit visible edges to recent or highest-volume flows.
- **Graph edges must show `event` info**:
  - Use edge labels OR tooltips (on hover) to show event names
  - If rendering is crowded, allow toggle between page-only vs. page+event view
- **Enable click-to-expand**:
  - Clicking a node expands a detail panel (session flows + metrics)
  - Use expandable containers, not modals
- **Support flow filtering**:
  - Filter fragile flows by length (e.g. 2, 3, or 4-node paths)
  - Optionally filter by minimum users affected
- **Maintain consistent coloring**:
  - 🔴 = top 10% WSJF
  - 🟠 = top 20%
  - ⚪ = neutral/low-friction nodes
- **Use collapsible tooltips**:
  - Every metric should be hoverable or come with a (?) icon
- **Avoid animation or latency-heavy graph rendering**:
  - Precompute node positions to avoid re-layout on every render
  - Use NetworkX layouts → export as JSON → render with `pyvis` or `streamlit-agraph`

- Only top nodes shown: filter top 10–20% WSJF by default
- Edge events must be visible: label or hover tooltip
- Clicking node expands detail view:
    Includes session flow list + summary metrics
- Flow filters:
    Min chokepoint count, min user loss, segment/tag
- Consistent visuals: do not mix themes or style formats
- Tooltips for all metrics: via hover or (?) icons
- Avoid layout lag:
    Precompute layout JSON if needed (esp. for 100+ nodes)


## ❌ Strict Prohibitions:
- ❌ No recomputation of WSJF or betweenness in `dashboard.py`
- ❌ Do not show all nodes or edges — overloads PMs with noise
- ❌ No color styles unless defined in `friction_node_map.json`
- ❌ No raw session tables — only aggregate insights

| Area        | Don’t Do This                                  |
| ----------- | ---------------------------------------------- |
| Computation | ❌ Recompute WSJF or betweenness                |
| Graph       | ❌ Show full user graph with no filtering       |
| Visuals     | ❌ Use light theme or inconsistent fonts/colors |
| UX          | ❌ Display raw session tables or CSVs directly  |

## 🎨 Dark Mode Graph Guidelines:

- Background: `#0D1117` or `#1A1F2B`
- Node colors:
  - 🔴 (top 10%): `#FF5C5C`
  - 🟠 (top 20%): `#F4A261`
  - ⚪ (neutral): `#CBD5E1`
- Edge colors: semi-transparent `#94A3B8`
- Font: use light sans-serif (e.g., `'Inter', sans-serif`) in white or gray `#E2E8F0`
- Tooltip boxes: darker background with light text
- Optional: glow/shadow for top nodes to create focus

If using `pyvis`, configure:
```python
net.set_options("""
const options = {
  "nodes": {
    "font": {"color": "#E2E8F0"},
    "color": {"highlight": {"border": "#FFFFFF"}}
  },
  "edges": {
    "color": "rgba(148, 163, 184, 0.6)",
    "smooth": false
  },
  "physics": false,
  "layout": {"improvedLayout": true}
}
""")

## ✅ Streamlit Testing Guidance (L2.x):
| Test Case                        | Description                                 |
| -------------------------------- | ------------------------------------------- |
| `test_dashboard_loads()`         | Confirms Streamlit loads with dark theme    |
| `test_friction_table_renders()`  | Verifies event\_chokepoints.csv is rendered |
| `test_graph_nodes_styled()`      | Top WSJF nodes have dark mode styling       |
| `test_tooltip_text_appears()`    | Hover tooltips appear for metric columns    |
| `test_click_drilldown_reveals()` | Clicking node expands detail pane           |


## 💡 UX Success Criteria (MVP):
PM can answer: “Which flows leak users and why?”
Top 10% friction nodes highlighted in 🔴
At least one actionable fragile flow shown
Tooltips explain every metric without jargon
Dashboard looks polished and readable in dark mode