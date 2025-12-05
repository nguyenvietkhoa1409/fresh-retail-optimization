# config/settings.py
import os

BASE_DIR_SETUP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE_DIR = os.path.join(BASE_DIR_SETUP, "data", "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = HF_CACHE_DIR

class ProjectConfig:
    BASE_DIR = BASE_DIR_SETUP
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "data", "artifacts", "part1")
    
    SEED = 2025
    HF_DATASET = "Dingdong-Inc/FreshRetailNet-50K"
    HOURS = list(range(6, 22))
    KEEP_SUBSET = True
    VOLUME_SHARE = 0.50
    PAIR_LIMIT = 200 
    
    # --- ECONOMICS (REVISED FOR TRADEOFF) ---
    ECONOMICS_MODE = "proxy"
    # High holding cost (0.5% daily) to punish U=5 (Hoarding)
    DAILY_HOLDING_RATE_PCT = 0.5 
    
    # [RESTORED] Missing config for Inventory Policy
    # Map Out-of-Stock rate to Target Service Level (Tau)
    # (Low OOS -> High Service Target)
    TAU_BY_OOS = [(0.00, 0.10, 0.95), (0.10, 0.30, 0.85), (0.30, 0.50, 0.75), (0.50, 1.01, 0.60)]
    
    CENTER_LAT = 31.2304
    CENTER_LON = 121.4737
    STORE_RADIUS_KM = (1, 15)
    SPEED_KMPH = 35.0 
    DRIVING_HOURS_PER_DAY = 10.0

    
    # [High Contrast Zones]
    SUPPLIER_ZONES = [
        (2, 5, 50, 1.60, 3.0, "Wholesaler_Near"),  # Very Expensive, Old
        (2, 50, 150, 1.00, 1.0, "Distributor_Mid"), 
        (2, 150, 400, 0.40, 0.0, "Farm_Far")       # Very Cheap, Fresh
    ]
    
    PRODUCT_CATEGORIES = [
        (101, 1, "Fresh Strawberries", 0.40, 0.0018),
        (102, 1, "Fresh Blueberries", 0.38, 0.0016),
        (201, 2, "Whole Milk", 0.15, 0.0010),
        (202, 2, "Eggs", 0.12, 0.0014),
    ]
    SHELF_LIFE_BY_CAT = {1: 7, 2: 12} 
    PRICE_RANGE_BY_PRODUCT = {101: (4.5, 7.0), 102: (6.0, 9.0), 201: (1.1, 1.6), 202: (1.8, 3.0)}
    
    HOLDING_COST_MULTIPLIER = 5.0 
    DEMAND_RANGE_BY_PRODUCT = {101: (40, 160), 102: (40, 140), 201: (160, 400), 202: (100, 300)}
    
    STORE_OPEN_WINDOW = (360, 600)    
    SUPPLIER_OPEN_WINDOW = (420, 960) 
    WAREHOUSE_WINDOW = (240, 1560)    
    
    SERVICE_TIME_STORE_MINS = 20
    SERVICE_TIME_SUPPLIER_MINS = 45
    SERVICE_TIME_CROSSDOCK_MINS = 90

    OUT_DIR_PART2 = os.path.join(BASE_DIR, "data", "artifacts", "part2")
    OUT_DIR_FORECAST = os.path.join(BASE_DIR, "data", "artifacts", "forecasting", "fixed")
    OUT_DIR_DIAGNOSTICS = os.path.join(BASE_DIR, "data", "artifacts", "diagnostics")
    
    TARGET_SUPPLY_DEMAND_RATIO = 2.0 
    MOQ_RANGE_UNITS = (1, 8)
    UNIT_WEIGHT = {101: 0.25, 102: 0.125, 201: 1.0, 202: 0.06}
    MIN_STD_FRACTION = 0.10
    
    OUT_DIR_PROCUREMENT = os.path.join(BASE_DIR, "data", "artifacts", "procurement")
    PROCURE_REVIEW_DAYS = 7.0
    SERVICE_LEVEL = 0.95 
    SHORTAGE_COST = 500.0
    
    # [CRITICAL FIX] Drastically reduce Fixed Cost so U=2 is viable
    FIXED_ORDER_COST = 5.0 
    
    TRANSPORT_COST_PER_KG_KM = 0.015 
    MAX_SOLVE_TIME_S = 900
    GAP_REL = 0.02
    ROUND_Q_TO_INT = True
    ALLOW_SHORTAGE = True
    VERBOSE_SOLVER = True
    
    OUT_DIR_LOGISTICS = os.path.join(BASE_DIR, "data", "artifacts", "vrp_route_maps")
    VRP_MAX_ROUTE_DISTANCE_KM = 600 
    VRP_SEARCH_TIME_LIMIT_SEC = 60
    VEHICLE_FLEET_DEFINITIONS = [
        {"type": "Small",  "capacity": 1000,   "fixed_cost": 300.0,  "cost_km": 0.5, "count": 15},
        {"type": "Medium", "capacity": 3000,   "fixed_cost": 500.0, "cost_km": 0.8, "count": 10},
        {"type": "Large",  "capacity": 8000,   "fixed_cost": 900.0, "cost_km": 1.2, "count": 5},
    ]
    VEHICLES_CHECK = [("Small", 1000.0), ("Medium", 3000.0), ("Large", 8000.0)]

    FRESHNESS_PENALTY_PER_DAY = 0.05 
    OUT_DIR_ANALYSIS = os.path.join(BASE_DIR, "data", "artifacts", "analysis_report")
    
    SEQ_LEN = 28; HORIZON = 7
    MAX_PAIRS = 1000; MIN_DAYS_PAIR = 90
    LGB_PARAMS = {"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 2025}

    # [SCENARIO MATRIX]
    STRATEGIC_SCENARIOS = [
        {"name": "Hyper-Fresh", "p": 2, "u": 2, "desc": "High Cost, Max Freshness"},
        {"name": "Local-Batch", "p": 2, "u": 5, "desc": "Local Sourcing, Min Logistics"},
        {"name": "Balanced", "p": 3, "u": 3, "desc": "Mid-range Sourcing"},
        {"name": "Bulk-Farm", "p": 4, "u": 3, "desc": "Farm Sourcing, Optimized Logistics"},
        {"name": "JIT-Farm", "p": 5, "u": 2, "desc": "Farm Sourcing, High Frequency"}
    ]