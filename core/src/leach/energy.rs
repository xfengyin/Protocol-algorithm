//! Energy consumption model for WSN

use serde::{Deserialize, Serialize};

/// First-order radio model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyModel {
    /// Electronics energy (nJ/bit)
    pub e_elec: f64,
    /// Free space amplifier (pJ/bit/m^2)
    pub e_fs: f64,
    /// Multi-path amplifier (pJ/bit/m^4)
    pub e_mp: f64,
    /// Data aggregation energy (nJ/bit)
    pub e_da: f64,
    /// Threshold distance (m)
    pub d0: f64,
    /// Packet size (bits)
    pub packet_size: f64,
}

impl Default for EnergyModel {
    fn default() -> Self {
        Self {
            e_elec: 50.0,      // 50 nJ/bit
            e_fs: 10.0,        // 10 pJ/bit/m^2
            e_mp: 0.0013,      // 0.0013 pJ/bit/m^4
            e_da: 5.0,         // 5 nJ/bit
            d0: 87.0,          // threshold distance
            packet_size: 4000.0, // 4000 bits
        }
    }
}

impl EnergyModel {
    /// Create custom energy model
    pub fn new(e_elec: f64, e_fs: f64, e_mp: f64, e_da: f64, packet_size: f64) -> Self {
        let d0 = (e_fs / e_mp).sqrt();
        Self {
            e_elec,
            e_fs,
            e_mp,
            e_da,
            d0,
            packet_size,
        }
    }

    /// Energy to transmit packet over distance d
    pub fn tx_energy(&self, d: f64) -> f64 {
        let e_amp = if d < self.d0 {
            self.e_fs * d * d
        } else {
            self.e_mp * d.powi(4)
        };
        (self.e_elec + e_amp) * self.packet_size
    }

    /// Energy to receive packet
    pub fn rx_energy(&self) -> f64 {
        self.e_elec * self.packet_size
    }

    /// Energy for data aggregation
    pub fn da_energy(&self, count: usize) -> f64 {
        self.e_da * self.packet_size * (count as f64)
    }

    /// Energy for cluster head to receive and aggregate from members
    pub fn ch_energy(&self, member_count: usize, avg_distance: f64) -> f64 {
        // Receive from members
        let rx = member_count as f64 * self.rx_energy();
        // Aggregate data
        let da = self.da_energy(member_count);
        // Transmit to base station
        let tx = self.tx_energy(avg_distance);
        rx + da + tx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model() {
        let model = EnergyModel::default();
        assert_eq!(model.e_elec, 50.0);
        assert_eq!(model.e_fs, 10.0);
        assert!((model.d0 - 87.0).abs() < 0.1);
    }

    #[test]
    fn test_tx_energy() {
        let model = EnergyModel::default();
        let e1 = model.tx_energy(50.0);  // Free space
        let e2 = model.tx_energy(100.0); // Multi-path
        assert!(e2 > e1);
    }

    #[test]
    fn test_rx_energy() {
        let model = EnergyModel::default();
        let e_rx = model.rx_energy();
        assert_eq!(e_rx, 50.0 * 4000.0); // e_elec * packet_size
    }
}
