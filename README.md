# Fresh Retail Supply Chain Optimization
## End-to-End E-Procurement & Inventory Management Framework

### 1. Introduction

This project addresses the complex challenges of **Freshness-Retail Supply Chains**, specifically focusing on the high perishability of goods and demand volatility. Traditional logistics models often fail to account for the degradation of fresh produce quality combined with the strict delivery windows required by retail stores.

We have developed a comprehensive **E-Logistics Optimization Framework** that integrates the supply chain workflow from demand data recovery to final last-mile delivery. The system utilizes the **Freshretailness50K** dataset to simulate a realistic retail environment with a structure of **Suppliers $\rightarrow$ Center Warehouse $\rightarrow$ Stores**.

**Key Objectives:**
* Mitigate the "Censored Demand" problem (unobserved lost sales) in retail data.
* Improve short-term demand forecasting accuracy for perishable goods.
* Optimize procurement strategies by balancing **Cost**, **Freshness**, and **Logistics** constraints.
* Solve the integrated Supplier Selection + Order Quantity + Vehicle Routing Problem (VRP).

---

### 2. Core Modules & Technical Framework

The project is structured into a sequential workflow where the output of one module serves as the critical input for the next.

#### Phase 1: Demand Recovery (Data Engineering)
* **Problem:** Retail sales data often represents "sales" rather than "demand." If an item is out of stock, demand is censored.
* **Solution:** A statistical reconstruction module estimates true demand from historical sales and inventory levels.
* **Metric:** Mean Absolute Percentage Error (MAPE) and Weighted APE (WAPE) on reconstructed vs. synthetic ground truth.

#### Phase 2: Demand Prediction (Forecasting)
* **Problem:** Accurate short-term forecasting (1-7 days) is critical for perishable inventory planning.
* **Method:** A machine learning-based forecasting engine (LightGBM) trained on the reconstructed demand data.
* **Scope:** Multi-horizon forecasting (H1 to H7) to support dynamic procurement planning.

#### Phase 3: Inventory Planning
* **Function:** Determines optimal inventory levels and safety stocks based on predicted demand and perishable shelf-life constraints.
* **Output:** Generates `Order Requirements` for the procurement phase.

#### Phase 4: Integrated Procurement & Logistics (Optimization)
* **The Core Engine:** This module solves the multi-objective optimization problem:
    1.  **Supplier Selection:** Which supplier offers the best trade-off between price and freshness?
    2.  **Order Quantity:** How much to order to meet demand without high spoilage?
    3.  **Logistics (VRP):** Optimizing delivery routes from suppliers to the warehouse/stores using distinct strategies (e.g., Cross-docking vs. Local Sourcing).

---

### 3. Results & Evaluation

Our framework was evaluated using a rigorous baseline comparison and sensitivity analysis. 

#### A. Demand Modeling Performance
The reconstruction and forecasting modules provide a solid foundation for the optimization engine.

| Metric | Demand Reconstruction | Demand Forecasting (Overall) |
| :--- | :--- | :--- |
| **WAPE** | **27.38%** | **36.19%** |
| **RMSE** | 0.35 | 2.18 |

#### B. Procurement Strategy Optimization (Scenario Analysis)
We simulated different procurement strategies to find the optimal balance between operational costs and freshness.

| Strategy | Description | Total Daily Cost ($) | Freshness Penalty | Logistics Cost |
| :--- | :--- | :--- | :--- | :--- |
| **Hyper-Fresh** | High Cost, Max Freshness | 50,257 | **51.9** | 1,404 |
| **Local-Batch** | Local Sourcing, Min Logistics | 47,591 | 116.1 | 925 |
| **Balanced** | Mid-range Sourcing | 46,016 | 69.4 | 935 |
| **JIT-Farm** | Farm Sourcing, High Freq | 43,610 | 43.7 | 1,404 |
| **Bulk-Farm (Proposed)** | **Farm Sourcing, Opt. Logistics** | **39,693** | **58.2** | **934** |

#### C. Comprehensive Baseline Comparison

We benchmarked our proposed **Bulk-Farm** model against a wide range of baselines, divided into Operational Strategies (Naive/Industry) and Theoretical/Ablation studies.

**Table 1: Operational Baselines (Naive & Industry Standards)**

| Strategy Name | Category | Daily Cost ($) | Gap to Optimal | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Proposed (Bulk-Farm)** | **Proposed** | **39,693** | **-** | **Integrated optimization (This work)** |
| Single-Tier (Regional) | Industry | 43,133 | +8.6% | Restrict to regional suppliers only |
| Random Assignment | Naive | 48,655 | +22.5% | Random supplier selection (worst case) |
| Single-Tier (Local) | Industry | 58,735 | +48.0% | Restrict to local suppliers only |
| Nearest Supplier | Naive | 61,965 | +56.1% | Always choose closest supplier (Myopic) |
| Cheapest Price | Naive | 64,003 | +61.2% | Lowest unit price only (Ignores distance) |
| Single-Tier (Farm) | Industry | 64,003 | +61.2% | Restrict to farm suppliers only |
| Equal Allocation | Industry | 129,904 | +227% | Split orders among top 3 suppliers (Diversification) |

**Table 2: Ablation Studies & Theoretical Bounds**

| Strategy Name | Category | Daily Cost ($) | Description |
| :--- | :--- | :--- | :--- |
| **No Fixed Costs** | Ablation | 21,760 | **Theoretical Lower Bound.** Ignoring setup/logistics fixed costs (Unrealistic). |
| **EOQ Policy** | Academic | 22,959 | Classic Economic Order Quantity (Textbook). Ignores complex constraints. |
| **No Capacity Limits** | Ablation | 26,756 | Assumes unlimited supplier capacity. |
| **No Freshness Penalty**| Ablation | 37,297 | Optimization without freshness consideration (Cheaper but lower quality). |

**Key Evaluation Insights:**
1.  **Efficiency:** The proposed model reduces daily costs by **69.4%** compared to the worst baseline (Equal Allocation) and outperforms the closest industry standard (Regional Sourcing) by **~8.6%**.
2.  **The "Cheapest Price" Trap:** Simply choosing the cheapest supplier (or Farm-only) results in high costs ($64k) due to inefficient logistics and high transport distances, proving that price alone is a poor indicator of total landed cost.
3.  **Logistics Impact:** The "Nearest Supplier" strategy performs poorly ($61k) because closest suppliers often have higher unit prices, failing to offset the transport savings.
4.  **Optimality:** Our result ($39k) is significantly higher than the theoretical lower bound ($21k - No Fixed Costs), effectively quantifying the "Price of Logistics" and operational reality in the supply chain.

---

### 4. Project Structure

The repository is organized as follows:

```text
fresh-retail-optimization/
├── config/                 # Configuration settings
├── data/
│   ├── artifacts/          # Generated plots, reports, and CSV results
│   └── ...                 # Input datasets
├── src/
│   ├── demand/             # Reconstruction and Forecasting modules
│   ├── inventory/          # Inventory planning logic
│   ├── optimization/       # Integrated solver (Cost evaluators, Logistics)
│   ├── data_pipeline/      # Data generators and preprocessors
│   └── utils/              # Helper functions (Geo-spatial, etc.)
├── run_all.py              # Main execution script
└── requirements.txt        # Dependencies
