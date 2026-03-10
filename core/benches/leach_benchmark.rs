//! LEACH Protocol Benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use protocol_algo_core::{LEACH, LEACHConfig, Simulation, Topology};

fn benchmark_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("LEACH Simulation");

    for node_count in [50, 100, 200, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(node_count),
            node_count,
            |b, &node_count| {
                let config = LEACHConfig {
                    p: 0.05,
                    rounds: 100,
                    initial_energy: 0.5,
                    base_station: (50.0, 150.0),
                    seed: 42,
                };

                let topology = Topology::random(100.0, 100.0, config.base_station);

                b.iter(|| {
                    let mut simulation = Simulation::new(
                        config.clone(),
                        topology.clone(),
                        *node_count,
                    );
                    black_box(simulation.run());
                });
            },
        );
    }

    group.finish();
}

fn benchmark_round(c: &mut Criterion) {
    let mut group = c.benchmark_group("LEACH Single Round");

    for node_count in [100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(node_count),
            node_count,
            |b, &node_count| {
                let config = LEACHConfig {
                    p: 0.05,
                    rounds: 1,
                    initial_energy: 0.5,
                    base_station: (50.0, 150.0),
                    seed: 42,
                };

                let topology = Topology::random(100.0, 100.0, config.base_station);

                b.iter(|| {
                    let mut simulation = Simulation::new(
                        config.clone(),
                        topology.clone(),
                        *node_count,
                    );
                    black_box(simulation.run());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_simulation, benchmark_round);
criterion_main!(benches);
