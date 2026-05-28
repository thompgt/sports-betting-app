# LineEdge - Multi-Bookmaker Arbitrage & EV Detection Engine

LineEdge is a high-performance quantitative sports betting analysis engine. It ingests live odds from multiple bookmakers, resolves disparate team/game entities, and uses advanced devigging models to identify positive Expected Value (+EV) and Arbitrage opportunities.

## 📊 Dashboard Preview
![LineEdge Dashboard](docs/assets/dashboard_preview.png)

## 🛠 System Architecture
```text
[ Data Ingestion ] ----> [ Entity Resolution ] ----> [ Quantitative Engine ]
      |                         |                           |
      | (Mock/Live Streams)     | (Exact/Fuzzy Matching)    | (No-Vig Power Method)
      v                         v                           v
[ SQLite Storage ] <-------------------------------- [ Edge Detection ]
      |
      +------> [ Streamlit Dashboard ]
```

## 🚀 Key Features
- **Real-Time Tracking:** Ingests live odds streams and flags market discrepancies instantly.
- **Entity Normalization:** Robust resolver that maps diverse naming conventions (e.g., "NY Rangers" vs "New York Rangers") to a canonical source of truth.
- **Advanced Math:** Implements the **No-Vig Power Method** for superior multi-way market devigging compared to traditional multiplicative models.
- **Deduplication & Caching:** Ensures that edges are only recorded once per market window unless a significant EV spike is detected.
- **Closing Line Value (CLV) Audit:** Built-in auditing layer to prove statistical edge against market closing lines.

## 🏁 Quick Start

### 1. Prerequisites
- Python 3.11+
- Playwright (for documentation capture)

### 2. Installation
```bash
git clone https://github.com/thompgt/sports-betting-app.git
cd sports-betting-app
pip install -r requirements.txt
playwright install chromium
```

### 3. Run Simulation & Backend
Populate the database with mock data to see the engine in action:
```bash
$env:PYTHONPATH = "."; python app/main_mock_run.py
```

### 4. Launch Dashboard
```bash
streamlit run app/ui/dashboard.py
```

### 5. Verify Installation
Run the full test suite (Entity Resolver, Math Utils, Ingestion, Storage):
```bash
$env:PYTHONPATH = "."; pytest
```

---
*Developed by thompgt. Built for high-frequency quantitative betting analysis.*
