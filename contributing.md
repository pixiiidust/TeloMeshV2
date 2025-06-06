# Contributing to TeloMesh

Thank you for your interest in contributing to TeloMesh! 🚀  
We're currently in an early **alpha prototype** stage, so contributions are especially valuable as we shape the foundation.

---

## 🔧 What We're Looking For

We're especially interested in help with:

- 🧱 **Modular enhancements** (layout modules, session flow parsers, percolation logic)
- 📊 **Visualization improvements** (Streamlit dashboard refinements, entropy graphs)
- 🤖 **Data ingestion** (synthetic session generators, connectors, n8n-compatible flows)
- 🧠 **ML/UX ideas** (funnel modeling, chokepoint detection, journey entropy)
- 📝 **Documentation** (readability, architecture sketches, example walkthroughs)

If you've found a bug or UX issue — please open an issue before submitting a PR.

---

## 📦 Setup

To get started locally:

1. Fork the repo → clone it to your machine
2. Install Python dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app

   ```bash
   streamlit run ui/dashboard.py
   ```

4. Generate sample data for testing

   ```bash
   python main.py --dataset test_dataset --users 100 --events 50
   ```

Optional:

* Configure `.env` for paths if using external datasets
* Enable n8n webhooks if live session flows are being tested

---

## 🧪 Testing

To run the test suite:

```bash
pytest
```

For specific test modules:

```bash
pytest tests/test_parse_sessions.py
pytest tests/test_event_chokepoints.py
```

All new features should include appropriate tests in the `/tests` directory.

---

## 📂 Folder Structure

```
TeloMesh/
├── data/                 ← Data generation and input storage
├── ingest/               ← Data ingestion and session parsing
├── analysis/             ← Analysis of user flows and friction points
├── ui/                   ← User interface dashboards
├── utils/                ← Utility scripts including Analytics Converter
├── outputs/              ← Generated output files by dataset
├── tests/                ← Test files for the project
├── logs/                 ← Logging and monitoring data
├── README.md             
└── CONTRIBUTING.md
```

---

## 🌿 Branch Strategy

* `main` - Production-ready code, stable releases
* `develop` - Integration branch for feature branches
* Feature branches - Named as `feature/descriptive-name`
* Bugfix branches - Named as `bugfix/issue-description`

Always create feature branches from `develop`, not `main`.

---

## 💻 Code Style

TeloMesh follows these style guidelines:

* PEP 8 for Python code style
* Use [Black](https://github.com/psf/black) for code formatting (line length 88)
* Use type hints where appropriate
* Use docstrings in Google style format
* Keep functions focused and under 50 lines when possible

---

## 🧪 Submitting Contributions

* Open a PR from your fork with a clear description
* Follow Python linting and formatting (e.g. Black, Ruff)
* Keep commits readable and modular (`feat:`, `fix:`, `refactor:`)
* Add inline comments if you're adding new logic-heavy modules
* Ensure all tests pass before submitting your PR
* Include documentation updates if you're changing user-facing features

---

## 🛟 Need Help?

Open an [issue](https://github.com/pixiiidust/TeloMeshV2/issues) or ping us in the Discussions tab. We'd love to hear your ideas.

---

Thanks again — together we can build smarter UX diagnostics!
