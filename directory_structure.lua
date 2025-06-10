# 📁 telomesh/ — MVP SLC Implementation (Simple + Lovable)

# Cursor Rule: Directory Hygiene Protocol
# ----------------------------------------
# Every file must:
# 1. Contain a clearly scoped set of related functions.
# 2. Be directly tied to a specific output or UI view.
# 3. Be covered by at least 1 test in /tests matching its prefix.
# 4. Follow the directory below — DO NOT create new folders without approval.
# 5. New files must register here and declare tests — else trigger the hygiene fix loop.

# ────────────────────────────────────────────────────────────────

📁 data/                     # SIMPLE STAGE — Events Generation
├── synthetic_event_generator.py     # Generate mock events
│   └── generate_synthetic_events(n_users: int, events_per_user: int) -> pd.DataFrame
├── synthetic_data.md        # Documentation about data generation
├── [dataset_name]/          # Dataset-specific input data
│   └── events.csv           # Raw event data for the dataset

📁 ingest/                     # SIMPLE STAGE — Event → Flow → Graph
├── parse_sessions.py         # Converts CSV → structured flows
│   └── parse_sessions(input_csv: str, output_csv: str) -> pd.DataFrame
│
├── build_graph.py            # Builds session path graph
│   └── build_graph(input_csv: str, output_graph: str, multi_graph: bool) -> nx.DiGraph or nx.MultiDiGraph
│
├── flow_metrics.py           # Validates sessions and graph
│   └── validate_sessions(session_flows_csv: str) -> dict
│   └── validate_graph(graph_pickle: str) -> dict
│   └── run_metrics() -> None
│
├── data_pipeline.md          # Documentation about data ingestion process

# ────────────────────────────────────────────────────────────────

📁 analysis/                   # LOVABLE STAGE — Metrics & Friction
├── event_chokepoints.py      # Compute friction metrics
│   └── compute_exit_rates(df: pd.DataFrame) -> pd.DataFrame
│   └── compute_betweenness(g: nx.DiGraph) -> Dict[str, float]
│   └── compute_wsjf_friction(exit_df, centrality_dict) -> pd.DataFrame
│   └── detect_fragile_flows(session_df: pd.DataFrame) -> pd.DataFrame
│   └── export_chokepoints(df, flow_df, node_map) -> None
│   └── convert_to_digraph(G: nx.MultiDiGraph) -> nx.DiGraph
│   └── compute_fractal_dimension(G: nx.DiGraph, max_radius: int = 10) -> float
│   └── compute_power_law_alpha(G: nx.DiGraph) -> float
│   └── detect_repeating_subgraphs(G: nx.DiGraph, max_length: int = 4) -> List[List[str]]
│   └── simulate_percolation(G: nx.DiGraph, ranked_nodes: List = None, threshold: float = 0.5, fast: bool = False) -> float
│   └── compute_fractal_betweenness(G: nx.DiGraph, repeating_subgraphs: List = None, centrality: Dict = None) -> Dict[str, float]
│   └── build_decision_table(G: nx.DiGraph, D: float, alpha: float, FB: Dict, threshold: float, chokepoints: pd.DataFrame, cc: float = None) -> pd.DataFrame
│   └── compute_clustering_coefficient(G: nx.DiGraph) -> float
│   └── main(input_flows: str, input_graph: str, input_graph_multi: str, output_dir: str, fast: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]
│
├── friction_analysis.md      # Documentation about friction analysis

# ────────────────────────────────────────────────────────────────

📁 ui/                         # LOVABLE STAGE — PM-Facing Dashboard
├── dashboard.py              # Streamlit dashboard
│   └── load_friction_data() -> Tuple[DataFrame, DataFrame, Dict]
│   └── render_friction_table(df: pd.DataFrame)
│   └── render_network_graph(net, height=1000) -> str
│   └── render_graph_heatmap(graph: nx.DiGraph, score_map: Dict[str, float])
│   └── render_flow_summaries(flow_df: pd.DataFrame)
│   └── render_transition_pairs(flow_df: pd.DataFrame)
│   └── load_advanced_metrics(dataset: str = None) -> Dict
│   └── render_fb_vs_wsjf_chart(metrics_data: Dict)
│   └── render_recurring_patterns(metrics_data: Dict)
│   └── render_percolation_collapse(metrics_data: Dict)
│   └── render_glossary_sidebar()
│   └── render_advanced_metrics_tab(metrics_data: Dict)
│   └── discover_datasets() -> List[str]
│   └── load_dataset_metadata(dataset_name: str) -> Dict
│   └── create_full_network(graph, score_map, top10_threshold, top20_threshold, physics_enabled, top50_threshold=None, layout_type="friction_levels") -> Network
│   └── create_filtered_network(graph, filtered_nodes, top10_threshold, top20_threshold, physics_enabled, top50_threshold=None, layout_type="friction_levels") -> Network
│
├── dashboard_backup.py       # Backup of the latest dashboard version
│
├── dashboard_components.md   # Documentation about dashboard components

# ────────────────────────────────────────────────────────────────

📁 utils/                      # Utility Tools
├── analytics_converter.py    # Convert from various analytics platforms
│   └── convert_data(input_file: str, output_file: str, format: str, telomesh_format: bool) -> pd.DataFrame
│   └── generate_sample_data(format: str, output_file: str) -> pd.DataFrame
│   └── map_to_telomesh_format(df: pd.DataFrame, format: str) -> pd.DataFrame
├── README.md                 # Comprehensive documentation for analytics converter

# ────────────────────────────────────────────────────────────────

📁 outputs/                    # Data generated by pipeline
├── [dataset_name]/           # Dataset-specific outputs
│   ├── session_flows.csv         # From parse_sessions.py
│   ├── user_graph.gpickle        # From build_graph.py
│   ├── user_graph_multi.gpickle  # From build_graph.py (for advanced analysis)
│   ├── event_chokepoints.csv     # From event_chokepoints.py
│   ├── high_friction_flows.csv   # From event_chokepoints.py
│   ├── friction_node_map.json    # From event_chokepoints.py
│   ├── decision_table.csv        # From event_chokepoints.py (UX recommendations)
│   ├── metrics.json              # From event_chokepoints.py (network metrics)
│   ├── dataset_info.json         # Dataset metadata
│   └── session_stats.log         # From flow_metrics.py
├── analysis_results.md           # Documentation about output files

# ────────────────────────────────────────────────────────────────

📁 logs/                      # Validation logs (legacy)
├── session_stats.log            # From flow_metrics.py (legacy)
├── metrics.json                 # From flow_metrics.py (legacy)

# ────────────────────────────────────────────────────────────────

📁 tests/                     # Enforces Cursor TDD loop
├── test_synthetic_events.py
├── test_parse_sessions.py
├── test_build_graph.py
├── test_flow_metrics.py
├── test_event_chokepoints.py    # Tests for all event_chokepoints.py functions
├── test_dashboard_ui.py
├── test_advanced_metrics_tab.py # Tests for advanced metrics visualizations
├── test_analytics_converter.py
├── test_dataset_organization.py  # Tests for dataset organization

# ────────────────────────────────────────────────────────────────

📁 logos/                     # Visual assets
├── telomesh logo.png           # Main logo
└── telomesh logo white.png     # White version for dark backgrounds

# ────────────────────────────────────────────────────────────────

main.py                       # Pipeline entrypoint
└── run_all_stages(dataset_name: str, users: int, events_per_user: int, input_file: str, multi: bool = False, fast: bool = False) -> None
└── create_dataset_directories(dataset_name: str) -> Tuple[str, str]
└── generate_dataset_metadata(dataset_name: str, data_dir: str, output_dir: str, users_count: int, events_count: int, sessions_count: int) -> Dict

