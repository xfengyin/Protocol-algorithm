use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

use protocol_algo_core::{LEACH, LEACHConfig, Simulation, Topology};

/// Application state
#[derive(Clone)]
struct AppState {
    // Shared state for simulations
}

/// Simulation request
#[derive(Debug, Deserialize)]
struct SimulationRequest {
    nodes: usize,
    rounds: usize,
    p: f64,
    area: f64,
    seed: Option<u64>,
}

/// Simulation response
#[derive(Debug, Serialize)]
struct SimulationResponse {
    success: bool,
    simulation_id: String,
    rounds: usize,
    initial_nodes: usize,
    final_alive: usize,
    survival_rate: f64,
}

/// Health check endpoint
async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "version": "2.0.0"
    }))
}

/// Create and run simulation
async fn create_simulation(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<SimulationRequest>,
) -> impl IntoResponse {
    let config = LEACHConfig {
        p: req.p,
        rounds: req.rounds,
        initial_energy: 0.5,
        base_station: (req.area / 2.0, req.area * 1.5),
        seed: req.seed.unwrap_or(42),
    };

    let topology = Topology::random(req.area, req.area, config.base_station);
    let mut simulation = Simulation::new(config, topology, req.nodes);
    let states = simulation.run();

    let final_state = states.last().unwrap();
    let survival_rate = (final_state.alive_nodes as f64 / req.nodes as f64) * 100.0;

    let response = SimulationResponse {
        success: true,
        simulation_id: format!("sim_{}", uuid::Uuid::new_v4()),
        rounds: req.rounds,
        initial_nodes: req.nodes,
        final_alive: final_state.alive_nodes,
        survival_rate,
    };

    (StatusCode::OK, Json(response))
}

/// Get simulation results
async fn get_simulation(
    _simulation_id: String,
) -> impl IntoResponse {
    // TODO: Implement simulation result retrieval
    Json(serde_json::json!({
        "message": "Not implemented yet"
    }))
}

/// Get network visualization data
async fn get_network_viz() -> impl IntoResponse {
    // TODO: Return network topology data for D3.js visualization
    Json(serde_json::json!({
        "nodes": [],
        "links": [],
        "cluster_heads": []
    }))
}

/// Create router
fn create_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/api/health", get(health))
        .route("/api/simulations", post(create_simulation))
        .route("/api/simulations/:id", get(get_simulation))
        .route("/api/viz/network", get(get_network_viz))
        .layer(cors)
        .nest_service("/", ServeDir::new("dist"))
        .with_state(state)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state = Arc::new(AppState {});
    let router = create_router(state);

    let addr = "0.0.0.0:3000";
    tracing::info!("🚀 Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, router).await.unwrap();
}
