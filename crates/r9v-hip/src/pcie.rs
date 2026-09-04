// SPDX-License-Identifier: Apache-2.0
//! Linux sysfs PCIe path discovery and bottleneck analysis (Spec 5 §3, Spec 14 §2, §3).

use std::fs;
use std::path::Path;

use crate::device::PciBdf;
use crate::error::{HipError, Result};

/// Computes the theoretical one-direction PCIe payload bandwidth in gigabytes per second (GB/s).
///
/// Accounts for PCIe line encoding overhead:
/// - PCIe Gen 1 (2.5 GT/s) and Gen 2 (5.0 GT/s) use 8b/10b encoding (80% efficiency).
/// - PCIe Gen 3 (8.0 GT/s) through Gen 6 (64.0 GT/s) use 128b/130b encoding (~98.46% efficiency).
pub fn pcie_payload_bandwidth_gbps(speed_gts: f64, width: u32) -> f64 {
    let efficiency = if speed_gts <= 5.0 { 0.8 } else { 128.0 / 130.0 };
    (speed_gts * width as f64 * efficiency) / 8.0
}

/// A single PCIe link hop along the path between the device endpoint and the root port.
#[derive(Debug, Clone, PartialEq)]
pub struct PciLinkHop {
    /// PCI BDF address of this bridge or endpoint.
    pub bdf: PciBdf,
    /// Currently negotiated link speed in GT/s, if available.
    pub current_speed_gts: Option<f64>,
    /// Currently negotiated link width in lanes (e.g. 1, 2, 4, 8, 16), if available.
    pub current_width: Option<u32>,
    /// Maximum supported link speed in GT/s, if available.
    pub max_speed_gts: Option<f64>,
    /// Maximum supported link width in lanes, if available.
    pub max_width: Option<u32>,
}

impl PciLinkHop {
    /// Computes the current negotiated payload capacity in GB/s, if speed and width are available.
    pub fn current_payload_bandwidth_gbps(&self) -> Option<f64> {
        match (self.current_speed_gts, self.current_width) {
            (Some(speed), Some(width)) if speed > 0.0 && width > 0 => {
                Some(pcie_payload_bandwidth_gbps(speed, width))
            }
            _ => None,
        }
    }

    /// Computes the maximum payload capacity in GB/s, if max speed and width are available.
    pub fn max_payload_bandwidth_gbps(&self) -> Option<f64> {
        match (self.max_speed_gts, self.max_width) {
            (Some(speed), Some(width)) if speed > 0.0 && width > 0 => {
                Some(pcie_payload_bandwidth_gbps(speed, width))
            }
            _ => None,
        }
    }

    /// Configured path capacity in GB/s. Maximum speed avoids treating an idle
    /// power-state downshift as topology, while negotiated width catches lane
    /// allocation, bifurcation and degraded training.
    pub fn capacity_payload_bandwidth_gbps(&self) -> Option<f64> {
        self.capacity_link()
            .map(|(speed, width)| pcie_payload_bandwidth_gbps(speed, width))
    }

    /// Speed/width pair used for configured-capacity classification.
    pub fn capacity_link(&self) -> Option<(f64, u32)> {
        let speed = self.max_speed_gts.or(self.current_speed_gts)?;
        let width = self.current_width.or(self.max_width)?;
        (speed > 0.0 && width > 0).then_some((speed, width))
    }

    /// Speed used for stable capacity classification.
    pub fn capacity_speed_gts(&self) -> Option<f64> {
        self.capacity_link().map(|(speed, _)| speed)
    }

    /// Width used for stable capacity classification.
    pub fn capacity_width(&self) -> Option<u32> {
        self.capacity_link().map(|(_, width)| width)
    }

    /// Returns the PCIe generation (1-6) corresponding to `current_speed_gts`, if known.
    pub fn current_generation(&self) -> Option<u32> {
        self.current_speed_gts.and_then(speed_to_generation)
    }

    /// Returns the PCIe generation (1-6) corresponding to `max_speed_gts`, if known.
    pub fn max_generation(&self) -> Option<u32> {
        self.max_speed_gts.and_then(speed_to_generation)
    }
}

fn speed_to_generation(speed: f64) -> Option<u32> {
    if (speed - 2.5).abs() < 0.1 {
        Some(1)
    } else if (speed - 5.0).abs() < 0.1 {
        Some(2)
    } else if (speed - 8.0).abs() < 0.1 {
        Some(3)
    } else if (speed - 16.0).abs() < 0.1 {
        Some(4)
    } else if (speed - 32.0).abs() < 0.1 {
        Some(5)
    } else if (speed - 64.0).abs() < 0.1 {
        Some(6)
    } else {
        None
    }
}

/// Discovered PCIe topology path information from device endpoint up to root port.
#[derive(Debug, Clone, PartialEq)]
pub struct PciPathDiscovery {
    /// Device endpoint link hop.
    pub endpoint: PciLinkHop,
    /// Upstream ancestors between the device endpoint and the root port,
    /// ordered from nearest parent to root port.
    pub upstream_ancestors: Vec<PciLinkHop>,
    /// Configured path-capacity bottleneck: maximum speed with negotiated
    /// width at each hop.
    pub capacity_bottleneck: PciLinkHop,
    /// Current-state bottleneck for diagnostics only. It must not key a cached
    /// plan because an idle link may downshift.
    pub current_bottleneck: Option<PciLinkHop>,
}

impl PciPathDiscovery {
    /// Discovers the PCIe path topology for `bdf` by reading Linux sysfs at `sys_root`.
    ///
    /// Inspects the endpoint and walks every upstream PCI bridge up to the root port.
    /// Selects the bottleneck by payload capacity, never endpoint alone.
    /// Does not infer PCIe width from P2P.
    pub fn discover(sys_root: &Path, bdf: PciBdf) -> Result<Self> {
        let bdf_str = bdf.to_string();
        let direct_bus_path = sys_root.join("bus/pci/devices").join(&bdf_str);
        let endpoint_dir = if direct_bus_path.exists() {
            direct_bus_path
        } else {
            let alt = sys_root.join(&bdf_str);
            if alt.exists() {
                alt
            } else {
                return Err(HipError::SysfsError {
                    bdf: bdf_str,
                    path: direct_bus_path.display().to_string(),
                    details: "PCI device not found in sysfs".to_string(),
                });
            }
        };

        // Resolve symlink to get true parent hierarchy under /sys/devices/...
        let canon_endpoint =
            fs::canonicalize(&endpoint_dir).map_err(|error| HipError::SysfsError {
                bdf: bdf_str.clone(),
                path: endpoint_dir.display().to_string(),
                details: format!("cannot resolve PCI ancestry: {error}"),
            })?;

        let endpoint_hop = read_pci_link_hop(&canon_endpoint, bdf);
        require_capacity_link(&endpoint_hop, &bdf_str, &canon_endpoint)?;

        let mut upstream_ancestors = Vec::new();
        let mut cur = canon_endpoint.parent();
        while let Some(parent) = cur {
            if let Some(name) = parent.file_name().and_then(|n| n.to_str()) {
                if let Ok(ancestor_bdf) = PciBdf::parse(name) {
                    let hop = read_pci_link_hop(parent, ancestor_bdf);
                    if hop.current_speed_gts.is_some()
                        || hop.current_width.is_some()
                        || hop.max_speed_gts.is_some()
                        || hop.max_width.is_some()
                    {
                        require_capacity_link(&hop, &bdf_str, parent)?;
                        upstream_ancestors.push(hop);
                    }
                }
            }
            cur = parent.parent();
        }

        let hops = std::iter::once(&endpoint_hop).chain(upstream_ancestors.iter());
        let capacity_bottleneck = hops
            .clone()
            .filter_map(|hop| {
                hop.capacity_payload_bandwidth_gbps()
                    .map(|bandwidth| (hop, bandwidth))
            })
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(hop, _)| hop.clone())
            .ok_or_else(|| HipError::SysfsError {
                bdf: bdf_str,
                path: canon_endpoint.display().to_string(),
                details: "no PCI hop exposed a complete speed/width pair".to_string(),
            })?;

        let current_bottleneck = hops
            .filter_map(|hop| {
                hop.current_payload_bandwidth_gbps()
                    .map(|bandwidth| (hop, bandwidth))
            })
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(hop, _)| hop.clone());

        Ok(Self {
            endpoint: endpoint_hop,
            upstream_ancestors,
            capacity_bottleneck,
            current_bottleneck,
        })
    }

    /// Returns the payload capacity of the bottleneck hop in GB/s.
    pub fn bottleneck_bandwidth_gbps(&self) -> Option<f64> {
        self.capacity_bottleneck.capacity_payload_bandwidth_gbps()
    }

    /// Returns the payload capacity of the endpoint hop in GB/s.
    pub fn endpoint_bandwidth_gbps(&self) -> Option<f64> {
        self.endpoint.capacity_payload_bandwidth_gbps()
    }

    /// Returns `true` if the bottleneck is located at an upstream bridge rather than the endpoint itself.
    pub fn is_bottlenecked_upstream(&self) -> bool {
        self.capacity_bottleneck.bdf != self.endpoint.bdf
    }

    /// Returns the lane width of the bottleneck hop.
    pub fn bottleneck_width(&self) -> Option<u32> {
        self.capacity_bottleneck.capacity_width()
    }

    /// Returns the transfer speed of the bottleneck hop in GT/s.
    pub fn bottleneck_speed_gts(&self) -> Option<f64> {
        self.capacity_bottleneck.capacity_speed_gts()
    }
}

fn require_capacity_link(hop: &PciLinkHop, endpoint_bdf: &str, path: &Path) -> Result<()> {
    if hop.capacity_link().is_some() {
        return Ok(());
    }
    Err(HipError::SysfsError {
        bdf: endpoint_bdf.to_owned(),
        path: path.display().to_string(),
        details: format!(
            "PCI hop {} did not expose a complete positive speed/width capacity pair",
            hop.bdf
        ),
    })
}

fn read_pci_link_hop(dir: &Path, bdf: PciBdf) -> PciLinkHop {
    let current_speed_gts = read_first_float(&dir.join("current_link_speed"));
    let current_width = read_first_int(&dir.join("current_link_width"));
    let max_speed_gts = read_first_float(&dir.join("max_link_speed"));
    let max_width = read_first_int(&dir.join("max_link_width"));

    PciLinkHop {
        bdf,
        current_speed_gts,
        current_width,
        max_speed_gts,
        max_width,
    }
}

fn read_first_float(path: &Path) -> Option<f64> {
    let text = fs::read_to_string(path).ok()?;
    parse_first_float(&text)
}

fn read_first_int(path: &Path) -> Option<u32> {
    let text = fs::read_to_string(path).ok()?;
    parse_first_int(&text)
}

pub(crate) fn parse_first_float(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    let mut start = None;
    for (i, c) in trimmed.char_indices() {
        if c.is_ascii_digit() {
            start = Some(i);
            break;
        }
    }
    let start = start?;
    let rest = &trimmed[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(rest.len());
    rest[..end].parse::<f64>().ok()
}

pub(crate) fn parse_first_int(text: &str) -> Option<u32> {
    let trimmed = text.trim();
    let mut start = None;
    for (i, c) in trimmed.char_indices() {
        if c.is_ascii_digit() {
            start = Some(i);
            break;
        }
    }
    let start = start?;
    let rest = &trimmed[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse::<u32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth_formula_exact_values() {
        // Gen1 x16 (2.5 GT/s, 80% efficiency) -> 4.0 GB/s
        let gen1_x16 = pcie_payload_bandwidth_gbps(2.5, 16);
        assert!((gen1_x16 - 4.0).abs() < 1e-4);

        // Gen2 x16 (5.0 GT/s, 80% efficiency) -> 8.0 GB/s
        let gen2_x16 = pcie_payload_bandwidth_gbps(5.0, 16);
        assert!((gen2_x16 - 8.0).abs() < 1e-4);

        // Gen4 x16 (16.0 GT/s, 128/130 efficiency) -> ~31.5077 GB/s
        let gen4_x16 = pcie_payload_bandwidth_gbps(16.0, 16);
        assert!((gen4_x16 - 31.50769).abs() < 1e-4);

        // Gen4 x4 (16.0 GT/s, 128/130 efficiency) -> ~7.8769 GB/s
        let gen4_x4 = pcie_payload_bandwidth_gbps(16.0, 4);
        assert!((gen4_x4 - 7.87692).abs() < 1e-4);

        // Gen5 x16 (32.0 GT/s, 128/130 efficiency) -> ~63.0154 GB/s
        let gen5_x16 = pcie_payload_bandwidth_gbps(32.0, 16);
        assert!((gen5_x16 - 63.01538).abs() < 1e-4);
    }

    #[test]
    fn test_parse_helpers() {
        assert_eq!(parse_first_float("16.0 GT/s PCIe"), Some(16.0));
        assert_eq!(parse_first_float("32.0 GT/s"), Some(32.0));
        assert_eq!(parse_first_float("5.0"), Some(5.0));
        assert_eq!(parse_first_float("invalid"), None);

        assert_eq!(parse_first_int("16"), Some(16));
        assert_eq!(parse_first_int("4\n"), Some(4));
        assert_eq!(parse_first_int("x16"), Some(16));
        assert_eq!(parse_first_int("none"), None);
    }
}
