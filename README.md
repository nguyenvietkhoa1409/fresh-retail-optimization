# Fresh Retail Supply Chain Optimization
## Integrated E-Procurement and Inventory Optimization Framework

### Executive Summary
This project presents a comprehensive end-to-end optimization framework for fresh retail supply chain management, integrating demand reconstruction, forecasting, inventory planning, procurement, and logistics optimization. The system addresses critical challenges in perishable goods management including **stockout-induced censored demand**, **multi-echelon sourcing decisions**, and **time-sensitive distribution constraints**.

---

### 1. Problem Statement & Current Situation

#### 1.1. Background
Fresh retail supply chains face unique operational challenges distinct from traditional retail environments (Aung & Chang, 2014). The perishability of products creates a fundamental tension between **overstocking** (leading to waste) and **understocking** (leading to lost sales and customer dissatisfaction). 

> *According to industry estimates, food waste in retail accounts for approximately 10-15% of purchased inventory (Mena et al., 2011), while stockouts can result in lost sales ranging from 3-4% of total revenue (Corsten & Gruen, 2003).*

#### 1.2. Core Challenges

**1.2.1. The Censored Demand Problem**
Traditional demand forecasting assumes observed sales data accurately represents true customer demand. However, in fresh retail environments with frequent stockouts, observed sales are systematically biased downward—a phenomenon known as "demand censoring" (Agrawal & Smith, 1996). 

Data analysis from the **FreshRetailNet-50K** dataset reveals that approximately **40-60%** of SKU-store-day observations exhibit some degree of intra-day stockout, making naive forecasting approaches fundamentally flawed.

*Mathematical Formulation:*
$$D_{observed} \leq D_{true}$$, 
$$D_{true} = D_{observed} + D_{lost}$$
Where $$D_{lost}$$ represents unobserved demand during stockout periods.

**1.2.2. Multi-Objective Procurement Decisions**
Fresh retail procurement requires simultaneous optimization across multiple conflicting objectives:
* **Cost Minimization:** Unit prices, fixed ordering costs, transportation costs.
* **Freshness Maximization:** Minimizing product age at delivery.
* **Service Level:** Meeting demand with high probability (typically 95%).
* **Operational Constraints:** Supplier capacities, Minimum Order Quantities (MOQ), Lead times.

**1.2.3. Coupled Procurement-Logistics Problem**
Traditional approaches treat procurement (supplier selection, order quantities) and logistics (vehicle routing, delivery scheduling) as sequential, independent problems. However, these decisions are fundamentally coupled:
* Supplier location affects transportation costs and delivery timing.
* Order consolidation impacts vehicle utilization.
* Delivery time windows constrain sourcing options.

#### 1.3. Research Gap
Existing literature addresses these challenges in isolation:
* *Demand reconstruction methods* (Weatherford & Pölt, 2002) focus on historical data calibration.
* *Newsvendor models* (Silver et al., 1998) optimize single-period inventory decisions.
* *Vehicle Routing Problems (VRP)* (Toth & Vigo, 2014) assume known demand and fixed supply points.

**This work integrates these components into a unified optimization framework, explicitly modeling the interdependencies between demand uncertainty, sourcing decisions, and distribution operations.**

---

### 2. Project Objectives

#### 2.1. Primary Objective
Develop an integrated decision support system that minimizes total supply chain costs while maintaining service level requirements for fresh retail operations, accounting for demand censorship, perishability constraints, and operational complexities.

#### 2.2. Specific Objectives

| Layer | ID | Objective Description |
| :--- | :--- | :--- |
| **Demand** | **O1.1** | Reconstruct true latent demand from censored sales observations with **WAPE < 30%**. |
| | **O1.2** | Generate multi-horizon demand forecasts (1-7 days) with horizon-specific accuracy. |
| | **O1.3** | Quantify demand uncertainty for downstream stochastic optimization. |
| **Supply** | **O2.1** | Design heterogeneous supplier network representing diverse sourcing strategies (local specialty, regional distributors, bulk wholesalers, farm-direct). |
| | **O2.2** | Calibrate supplier attributes (pricing, lead times, capacities, freshness degradation) to create realistic trade-offs. |
| **Optimization**| **O3.1** | Formulate and solve **Mixed-Integer Linear Program (MILP)** for multi-product, multi-supplier procurement under capacity, MOQ, and lead time constraints. |
| | **O3.2** | Solve two-echelon **Vehicle Routing Problem (VRP)** with time windows for inbound and outbound distribution. |
| | **O3.3** | Implement iterative feedback mechanism between procurement and logistics to escape local optima. |
| **Evaluation** | **O4.1** | Establish rigorous baseline comparison framework across four categories: naive heuristics, industry practices, academic benchmarks, ablation studies. |

---

### 3. Technical Framework & Methodology

The project is structured into a sequential workflow where the output of one module serves as the critical input for the next.

#### Phase 1: Demand Recovery (Data Engineering)
* **Input:** Historical sales transactions and inventory logs.
* **Method:** Statistical reconstruction to estimate $D_{lost}$ and un-censor the data.
* **Output:** A clean, "True Demand" dataset for training.

#### Phase 2: Demand Prediction (Forecasting)
* **Method:** Machine Learning engine (LightGBM) with multi-horizon capability (H1-H7).
* **Features:** Lagged features, calendar events, price elasticity, and reconstructed historical demand.

#### Phase 3: Inventory Planning
* **Function:** Determines optimal inventory levels and safety stocks.
* **Logic:** Dynamic safety stock calculation based on forecast error variance (RMSE) and target service levels (95%).

#### Phase 4: Integrated Procurement & Logistics (Optimization)
* **The Core Engine:** Solves the coupled problem defined in *Section 1.2.3*.
    1.  **Supplier Selection:** Trade-off between Price vs. Freshness.
    2.  **Order Quantity:** Optimal batch sizes respecting MOQs.
    3.  **Logistics (VRP):** Optimizing delivery routes from Suppliers $\rightarrow$ Center Warehouse $\rightarrow$ Stores.

---

### 4. Results & Evaluation

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

We benchmarked our proposed **Bulk-Farm** model against a wide range of baselines.

**Table 1: Operational Baselines (Naive & Industry Standards)**

| Strategy Name | Category | Daily Cost ($) | Gap to Optimal | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Proposed (Bulk-Farm)** | **Proposed** | **39,693** | **-** | **Integrated optimization (This work)** |
| Single-Tier (Regional) | Industry | 43,133 | +8.6% | Restrict to regional suppliers only |
| Random Assignment | Naive | 48,655 | +22.5% | Random supplier selection (worst case) |
| Single-Tier (Local) | Industry | 58,735 | +48.0% | Restrict to local suppliers only |
| Nearest Supplier | Naive | 61,965 | +56.1% | Always choose closest supplier (Myopic) |
| Cheapest Price | Naive | 64,003 | +61.2% | Lowest unit price only (Ignores distance) |
| Equal Allocation | Industry | 129,904 | +227% | Split orders among top 3 suppliers |

**Table 2: Ablation Studies & Theoretical Bounds**

| Strategy Name | Category | Daily Cost ($) | Description |
| :--- | :--- | :--- | :--- |
| **No Fixed Costs** | Ablation | 21,760 | **Theoretical Lower Bound.** Ignoring setup/logistics fixed costs. |
| **EOQ Policy** | Academic | 22,959 | Classic Economic Order Quantity (Textbook). |
| **No Freshness Penalty**| Ablation | 37,297 | Optimization without freshness consideration. |

**Key Evaluation Insights:**
1.  **Efficiency:** The proposed model reduces daily costs by **69.4%** compared to the worst baseline and outperforms the closest industry standard (Regional Sourcing) by **~8.6%**.
2.  **The "Cheapest Price" Trap:** Simply choosing the cheapest supplier results in high costs ($64k) due to inefficient logistics, proving that price alone is a poor indicator of total landed cost.
3.  **Optimality:** Our result ($39k) is significantly higher than the theoretical lower bound ($21k), effectively quantifying the "Price of Logistics" and operational reality.

---

### 5. Project Structure

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
