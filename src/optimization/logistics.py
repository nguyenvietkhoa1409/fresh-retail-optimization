# src/optimization/logistics.py
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils

class VRPSolver:
    """
    Generic VRP Solver hỗ trợ Central Warehouse Model.
    - Depot: Warehouse Coordinates.
    - Nodes: Suppliers (Inbound) hoặc Stores (Outbound).
    - Constraints: Capacity, Max Distance.
    """
    def __init__(self, depot_loc, stops_df, role="Inbound"):
        self.depot_loc = depot_loc  # (lat, lon) của kho trung tâm
        self.stops_df = stops_df    # DataFrame chứa các điểm dừng: ['lat', 'lon', 'demand_kg', 'id']
        self.role = role            # "Inbound" hoặc "Outbound"
        
        self.data = {}; self.manager = None; self.routing = None; self.solution = None
        
        # Load Fleet Definition
        self.vehicles = []
        for v_def in Cfg.VEHICLE_FLEET_DEFINITIONS:
            for _ in range(v_def['count']):
                self.vehicles.append(v_def)
        # Sắp xếp xe từ nhỏ đến lớn để solver ưu tiên xe nhỏ nếu fit
        self.vehicles.sort(key=lambda x: x['capacity'])

    def solve(self):
        self._create_data_model()
        self.manager = pywrapcp.RoutingIndexManager(len(self.data['distance_matrix']), self.data['num_vehicles'], self.data['depot'])
        self.routing = pywrapcp.RoutingModel(self.manager)
        
        # Add Constraints
        self._add_distance_dimension() # Bao gồm Max Route Distance
        self._add_capacity_dimension()
        self._set_vehicle_costs()
        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = Cfg.VRP_SEARCH_TIME_LIMIT_SEC
        
        self.solution = self.routing.SolveWithParameters(search_parameters)
        return self._extract_routes()

    def _create_data_model(self):
        # Node 0 luôn là Warehouse (Depot)
        locations = [self.depot_loc] 
        demands = [0]
        ids = ["WAREHOUSE"]
        
        for _, row in self.stops_df.iterrows():
            locations.append((row['lat'], row['lon']))
            # Slack 150% đã được loại bỏ vì Logic 5x demand an toàn hơn
            # Nhưng vẫn ceil để đảm bảo integer
            demands.append(int(math.ceil(row['demand_kg'])))
            ids.append(str(row['id']))
            
        n = len(locations)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    km = GeoUtils.haversine_km(locations[i][0], locations[i][1], locations[j][0], locations[j][1])
                    dist_matrix[i][j] = int(km * 1000) # Meters
                    
        self.data = {
            'locations': locations, 'demands': demands, 'ids': ids,
            'distance_matrix': dist_matrix, 'num_vehicles': len(self.vehicles),
            'vehicle_capacities': [v['capacity'] for v in self.vehicles], 'depot': 0
        }

    def _add_distance_dimension(self):
        def dist_cb(i, j):
            return int(self.data['distance_matrix'][self.manager.IndexToNode(i)][self.manager.IndexToNode(j)])
        cb_idx = self.routing.RegisterTransitCallback(dist_cb)
        
        # --- ÁP DỤNG RÀNG BUỘC MAX ROUTE DISTANCE (Realism) ---
        # Ngăn chặn xe chạy "World Tour". Ép buộc chia nhỏ lộ trình.
        max_dist_meters = Cfg.VRP_MAX_ROUTE_DISTANCE_KM * 1000
        self.routing.AddDimension(cb_idx, 0, max_dist_meters, True, "Distance")

    def _add_capacity_dimension(self):
        def dem_cb(i): return self.data['demands'][self.manager.IndexToNode(i)]
        cb_idx = self.routing.RegisterUnaryTransitCallback(dem_cb)
        self.routing.AddDimensionWithVehicleCapacity(cb_idx, 0, self.data['vehicle_capacities'], True, "Capacity")

    def _set_vehicle_costs(self):
        for i in range(self.data['num_vehicles']):
            v = self.vehicles[i]
            self.routing.SetFixedCostOfVehicle(int(v['fixed_cost']), i)
            def dist_cb(from_idx, to_idx):
                dm = self.data['distance_matrix'][self.manager.IndexToNode(from_idx)][self.manager.IndexToNode(to_idx)]
                return int((dm / 1000.0) * v['cost_km'])
            idx = self.routing.RegisterTransitCallback(dist_cb)
            self.routing.SetArcCostEvaluatorOfVehicle(idx, i)

    def _extract_routes(self):
        routes = []
        if not self.solution: return routes
        for i in range(self.data['num_vehicles']):
            index = self.routing.Start(i)
            if self.routing.IsEnd(self.solution.Value(self.routing.NextVar(index))): continue
            steps = []; load = 0; dist_m = 0; v = self.vehicles[i]
            
            while not self.routing.IsEnd(index):
                node = self.manager.IndexToNode(index)
                load += self.data['demands'][node]
                
                # Xác định loại node để vẽ cho đẹp
                node_type = "DEPOT" if node == 0 else ("SUPPLIER" if self.role == "Inbound" else "STORE")
                
                steps.append({
                    "node": node, "id": self.data['ids'][node],
                    "lat": self.data['locations'][node][0], "lon": self.data['locations'][node][1],
                    "type": node_type
                })
                prev = index; index = self.solution.Value(self.routing.NextVar(index))
                dist_m += self.data['distance_matrix'][self.manager.IndexToNode(prev)][self.manager.IndexToNode(index)]
            
            # Quay về Depot
            steps.append({"node": 0, "id": "WAREHOUSE", "lat": self.data['locations'][0][0], "lon": self.data['locations'][0][1], "type": "DEPOT"})
            dist_km = dist_m / 1000.0
            
            routes.append({
                "vehicle_id": i, "vehicle_type": v['type'], "capacity": v['capacity'],
                "role": self.role,
                "route_path": [s['id'] for s in steps], "steps": steps,
                "total_load_kg": load, "distance_km": dist_km,
                "cost": v['fixed_cost'] + (dist_km * v['cost_km']),
                "utilization_pct": round(load / v['capacity'] * 100, 1)
            })
        return routes

class RouteVisualizer:
    @staticmethod
    def plot_routes(routes, depot_loc, role):
        plt.figure(figsize=(10, 8))
        plt.scatter(depot_loc[1], depot_loc[0], c='black', s=300, marker='P', label='Central Warehouse', zorder=10)
        
        colors = plt.cm.get_cmap('tab10', len(routes))
        for idx, r in enumerate(routes):
            lats = [s['lat'] for s in r['steps']]; lons = [s['lon'] for s in r['steps']]
            label = f"{r['vehicle_type']} ({r['total_load_kg']}kg)"
            plt.plot(lons, lats, c=colors(idx), linewidth=2, linestyle='-', label=label, alpha=0.8)
            
            # Plot Stops
            stop_lats = lats[1:-1]; stop_lons = lons[1:-1]
            marker = '^' if role == "Inbound" else 'o'
            plt.scatter(stop_lons, stop_lats, c=colors(idx), s=60, marker=marker, edgecolors='white')
            
        title = "Inbound Logistics (Suppliers -> Warehouse)" if role == "Inbound" else "Outbound Logistics (Warehouse -> Stores)"
        plt.title(title)
        plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(Cfg.OUT_DIR_LOGISTICS, f"route_map_{role.lower()}.png")); plt.close()

class LogisticsManager:
    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_LOGISTICS, exist_ok=True)
        self.warehouse_loc = (Cfg.CENTER_LAT, Cfg.CENTER_LON)
    
    def run(self):
        print("\n[Logistics] Starting Centralized VRP (Inbound + Outbound)...")
        
        # Load Data
        agg_path = os.path.join(Cfg.OUT_DIR_PROCUREMENT, "procurement_summary_by_supplier_store.csv")
        sup_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers_harmonized_v2.csv")
        store_path = os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet")
        
        if not os.path.exists(agg_path): return
        df_agg = pd.read_csv(agg_path)
        df_sup = pd.read_csv(sup_path)
        df_store = pd.read_parquet(store_path)
        
        # 1. PREPARE INBOUND DATA (Suppliers -> Warehouse)
        inbound_demand = df_agg.groupby('supplier_id')['total_kg'].sum().reset_index()
        inbound_demand = inbound_demand.merge(df_sup[['supplier_id', 'sup_lat', 'sup_lon']], on='supplier_id')
        inbound_demand = inbound_demand.rename(columns={'supplier_id': 'id', 'sup_lat': 'lat', 'sup_lon': 'lon', 'total_kg': 'demand_kg'})
        
        print(f"  -> Solving Inbound VRP ({len(inbound_demand)} Suppliers)...")
        in_solver = VRPSolver(self.warehouse_loc, inbound_demand, role="Inbound")
        in_routes = in_solver.solve()
        if in_routes: RouteVisualizer.plot_routes(in_routes, self.warehouse_loc, "Inbound")
            
        # 2. PREPARE OUTBOUND DATA (Warehouse -> Stores)
        outbound_demand = df_agg.groupby('store_id')['total_kg'].sum().reset_index()
        df_store['store_id'] = df_store['store_id'].astype(str)
        outbound_demand['store_id'] = outbound_demand['store_id'].astype(str)
        
        store_locs = df_store.groupby('store_id').first()[['store_lat', 'store_lon']].reset_index()
        outbound_demand = outbound_demand.merge(store_locs, on='store_id')
        outbound_demand = outbound_demand.rename(columns={'store_id': 'id', 'store_lat': 'lat', 'store_lon': 'lon', 'total_kg': 'demand_kg'})
        
        print(f"  -> Solving Outbound VRP ({len(outbound_demand)} Stores)...")
        out_solver = VRPSolver(self.warehouse_loc, outbound_demand, role="Outbound")
        out_routes = out_solver.solve()
        if out_routes: RouteVisualizer.plot_routes(out_routes, self.warehouse_loc, "Outbound")

        # 3. SAVE
        all_routes = in_routes + out_routes
        
        # Lưu file chi tiết
        pd.DataFrame(all_routes).to_csv(os.path.join(Cfg.OUT_DIR_LOGISTICS, "vrp_routes_solution.csv"), index=False)
        
        # --- ĐOẠN CODE TẠO vrp_summary.csv BỊ THIẾU ---
        # Tính tổng chi phí toàn mạng lưới
        total_dist = sum(r['distance_km'] for r in all_routes)
        total_cost = sum(r['cost'] for r in all_routes)
        total_load = sum(r['total_load_kg'] for r in all_routes)
        
        # Tạo DataFrame tóm tắt
        summary_data = [{
            'Role': 'Total Network',
            'Routes_Count': len(all_routes),
            'Distance_KM': total_dist,
            'Load_KG': total_load,
            'Cost_USD': total_cost  # <--- IntegratedSolver CẦN CỘT NÀY
        }]
        
        pd.DataFrame(summary_data).to_csv(os.path.join(Cfg.OUT_DIR_LOGISTICS, "vrp_summary.csv"), index=False)
