"""
IoT Sensor Simulator for Remote Patient Monitoring.
Generates realistic vital sign streams seeded from real distributions.

Based on:
- MIMIC-III vital signs distributions (Johnson et al., 2016)
- PhysioNet vital sign databases (Goldberger et al., 2000)
- Clinical vital sign reference ranges
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, VitalSignsConfig, SimulationConfig


class AnomalyType(Enum):
    """Types of anomalies that can be injected."""
    NONE = 0
    BRADYCARDIA = 1  # Low heart rate
    TACHYCARDIA = 2  # High heart rate
    HYPOXEMIA = 3    # Low SpO2
    HYPOTENSION = 4  # Low blood pressure
    HYPERTENSION = 5 # High blood pressure
    HYPOTHERMIA = 6  # Low temperature
    HYPERTHERMIA = 7 # High temperature (fever)
    TACHYPNEA = 8    # High respiratory rate
    BRADYPNEA = 9    # Low respiratory rate
    MULTI_ORGAN = 10 # Multiple vital signs affected (sepsis-like)


@dataclass
class PatientProfile:
    """Individual patient baseline profile."""
    patient_id: int
    age: int
    gender: str
    baseline_hr: float
    baseline_spo2: float
    baseline_sbp: float
    baseline_dbp: float
    baseline_temp: float
    baseline_rr: float
    condition: str  # "healthy", "chronic", "acute"


class IoTSensorSimulator:
    """
    Simulates IoT sensor streams for multiple patients.
    
    Features:
    - Realistic vital sign distributions based on MIMIC-III
    - Circadian rhythm variations
    - Inter-patient variability
    - Sensor noise modeling
    - Configurable anomaly injection
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.vs_config = self.config.vital_signs
        self.sim_config = self.config.simulation
        
        np.random.seed(self.sim_config.random_seed)
        
        self.patients: Dict[int, PatientProfile] = {}
        self.anomaly_labels: Dict[int, List[Tuple[int, int, AnomalyType]]] = {}
        
    def _create_patient_profile(self, patient_id: int) -> PatientProfile:
        """
        Create a patient profile with individual baseline characteristics.
        Baselines vary ±1 std from population mean to simulate inter-patient variability.
        """
        age = np.random.randint(25, 85)
        gender = np.random.choice(["M", "F"])
        condition = np.random.choice(["healthy", "chronic", "acute"], p=[0.6, 0.3, 0.1])
        
        # Age and condition affect baselines
        age_factor = 1.0 + (age - 50) * 0.005  # Older patients tend to have higher BP
        condition_factor = {"healthy": 1.0, "chronic": 1.1, "acute": 1.2}[condition]
        
        profile = PatientProfile(
            patient_id=patient_id,
            age=age,
            gender=gender,
            baseline_hr=np.random.normal(
                self.vs_config.heart_rate_mean * condition_factor,
                self.vs_config.heart_rate_std * 0.5
            ),
            baseline_spo2=np.random.normal(
                self.vs_config.spo2_mean - (0 if condition == "healthy" else 1),
                self.vs_config.spo2_std * 0.3
            ),
            baseline_sbp=np.random.normal(
                self.vs_config.sbp_mean * age_factor,
                self.vs_config.sbp_std * 0.5
            ),
            baseline_dbp=np.random.normal(
                self.vs_config.dbp_mean * age_factor,
                self.vs_config.dbp_std * 0.5
            ),
            baseline_temp=np.random.normal(
                self.vs_config.temperature_mean,
                self.vs_config.temperature_std * 0.3
            ),
            baseline_rr=np.random.normal(
                self.vs_config.resp_rate_mean * condition_factor,
                self.vs_config.resp_rate_std * 0.5
            ),
            condition=condition
        )
        
        # Clip to physiological ranges
        profile.baseline_hr = np.clip(profile.baseline_hr, 55, 95)
        profile.baseline_spo2 = np.clip(profile.baseline_spo2, 94, 100)
        profile.baseline_sbp = np.clip(profile.baseline_sbp, 100, 140)
        profile.baseline_dbp = np.clip(profile.baseline_dbp, 60, 90)
        profile.baseline_temp = np.clip(profile.baseline_temp, 36.2, 37.4)
        profile.baseline_rr = np.clip(profile.baseline_rr, 12, 20)
        
        return profile
    
    def _get_circadian_factor(self, hour: float) -> Dict[str, float]:
        """
        Model circadian rhythm effects on vital signs.
        
        Based on: Smolensky et al. (2015). Diurnal and twenty-four hour patterning of
        human diseases: cardiac, vascular, and respiratory diseases, conditions, and syndromes.
        """
        # Heart rate and BP dip during sleep (night), peak in morning
        sleep_factor = np.cos(2 * np.pi * (hour - 3) / 24)  # Min at 3am
        
        return {
            "hr": 1.0 - 0.08 * sleep_factor,      # ±8% variation
            "sbp": 1.0 - 0.10 * sleep_factor,     # ±10% variation (nocturnal dipping)
            "dbp": 1.0 - 0.10 * sleep_factor,
            "temp": 1.0 - 0.01 * sleep_factor,    # ±0.3°C variation
            "rr": 1.0 - 0.05 * sleep_factor,      # ±5% variation
            "spo2": 1.0                           # SpO2 relatively stable
        }
    
    def _add_sensor_noise(self, value: float, sensor_type: str) -> float:
        """
        Add realistic sensor noise based on device specifications.
        
        Noise models based on typical wearable sensor accuracy:
        - PPG heart rate: ±2 bpm
        - Pulse oximeter: ±2%
        - Blood pressure cuff: ±3 mmHg
        - Temperature: ±0.1°C
        """
        noise_models = {
            "hr": np.random.normal(0, 2),
            "spo2": np.random.normal(0, 1),
            "sbp": np.random.normal(0, 3),
            "dbp": np.random.normal(0, 2),
            "temp": np.random.normal(0, 0.1),
            "rr": np.random.normal(0, 1)
        }
        return value + noise_models.get(sensor_type, 0)
    
    def _generate_anomaly_sequence(
        self,
        anomaly_type: AnomalyType,
        duration: int,
        severity: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Generate anomaly patterns that evolve over time.
        Anomalies have onset, peak, and resolution phases.
        """
        # Envelope: ramp up, sustain, ramp down
        onset = duration // 4
        sustain = duration // 2
        resolution = duration - onset - sustain
        
        envelope = np.concatenate([
            np.linspace(0, 1, onset),
            np.ones(sustain),
            np.linspace(1, 0, resolution)
        ])
        envelope = envelope[:duration] * severity
        
        anomaly_effects = {
            AnomalyType.BRADYCARDIA: {"hr": -30 * envelope},
            AnomalyType.TACHYCARDIA: {"hr": 50 * envelope},
            AnomalyType.HYPOXEMIA: {"spo2": -10 * envelope, "hr": 20 * envelope},
            AnomalyType.HYPOTENSION: {"sbp": -40 * envelope, "dbp": -20 * envelope, "hr": 15 * envelope},
            AnomalyType.HYPERTENSION: {"sbp": 50 * envelope, "dbp": 25 * envelope},
            AnomalyType.HYPOTHERMIA: {"temp": -3 * envelope, "hr": -10 * envelope},
            AnomalyType.HYPERTHERMIA: {"temp": 2.5 * envelope, "hr": 25 * envelope, "rr": 8 * envelope},
            AnomalyType.TACHYPNEA: {"rr": 15 * envelope, "spo2": -3 * envelope},
            AnomalyType.BRADYPNEA: {"rr": -8 * envelope, "spo2": -5 * envelope},
            AnomalyType.MULTI_ORGAN: {
                "hr": 40 * envelope,
                "sbp": -30 * envelope,
                "spo2": -8 * envelope,
                "temp": 2 * envelope,
                "rr": 10 * envelope
            }
        }
        
        return anomaly_effects.get(anomaly_type, {})
    
    def simulate_patient(
        self,
        patient_id: int,
        duration_hours: Optional[float] = None,
        inject_anomalies: bool = True
    ) -> pd.DataFrame:
        """
        Simulate vital sign stream for a single patient.
        
        Returns DataFrame with columns:
        - timestamp, patient_id
        - heart_rate, spo2, sbp, dbp, temperature, resp_rate
        - is_anomaly, anomaly_type
        """
        duration_hours = duration_hours or self.sim_config.simulation_hours
        
        # Create or get patient profile
        if patient_id not in self.patients:
            self.patients[patient_id] = self._create_patient_profile(patient_id)
        profile = self.patients[patient_id]
        
        # Calculate number of samples
        samples_per_hour = 3600 // self.sim_config.sampling_interval
        n_samples = int(duration_hours * samples_per_hour)
        
        # Initialize time series
        timestamps = pd.date_range(
            start="2024-01-01 00:00:00",
            periods=n_samples,
            freq=f"{self.sim_config.sampling_interval}s"
        )
        
        # Generate baseline signals with circadian variation
        data = {
            "timestamp": timestamps,
            "patient_id": [patient_id] * n_samples,
            "heart_rate": np.zeros(n_samples),
            "spo2": np.zeros(n_samples),
            "sbp": np.zeros(n_samples),
            "dbp": np.zeros(n_samples),
            "temperature": np.zeros(n_samples),
            "resp_rate": np.zeros(n_samples),
            "is_anomaly": np.zeros(n_samples, dtype=int),
            "anomaly_type": ["NONE"] * n_samples
        }
        
        for i in range(n_samples):
            hour = (timestamps[i].hour + timestamps[i].minute / 60)
            circadian = self._get_circadian_factor(hour)
            
            # Generate vital signs with circadian modulation and noise
            data["heart_rate"][i] = self._add_sensor_noise(
                profile.baseline_hr * circadian["hr"], "hr"
            )
            data["spo2"][i] = self._add_sensor_noise(
                profile.baseline_spo2 * circadian["spo2"], "spo2"
            )
            data["sbp"][i] = self._add_sensor_noise(
                profile.baseline_sbp * circadian["sbp"], "sbp"
            )
            data["dbp"][i] = self._add_sensor_noise(
                profile.baseline_dbp * circadian["dbp"], "dbp"
            )
            data["temperature"][i] = self._add_sensor_noise(
                profile.baseline_temp * circadian["temp"], "temp"
            )
            data["resp_rate"][i] = self._add_sensor_noise(
                profile.baseline_rr * circadian["rr"], "rr"
            )
        
        # Inject anomalies
        if inject_anomalies:
            data = self._inject_anomalies(data, n_samples, patient_id)
        
        # Clip to physiological limits
        data["heart_rate"] = np.clip(data["heart_rate"], 20, 220)
        data["spo2"] = np.clip(data["spo2"], 60, 100)
        data["sbp"] = np.clip(data["sbp"], 50, 250)
        data["dbp"] = np.clip(data["dbp"], 30, 150)
        data["temperature"] = np.clip(data["temperature"], 32, 43)
        data["resp_rate"] = np.clip(data["resp_rate"], 4, 50)
        
        return pd.DataFrame(data)
    
    def _inject_anomalies(
        self,
        data: Dict,
        n_samples: int,
        patient_id: int
    ) -> Dict:
        """Inject anomaly episodes into the vital sign stream."""
        anomaly_types = list(AnomalyType)[1:]  # Exclude NONE
        
        self.anomaly_labels[patient_id] = []
        
        i = 0
        while i < n_samples:
            if np.random.random() < self.sim_config.anomaly_rate:
                # Start an anomaly episode
                duration = np.random.randint(
                    self.sim_config.anomaly_duration_min,
                    self.sim_config.anomaly_duration_max
                )
                duration = min(duration, n_samples - i)
                
                anomaly_type = np.random.choice(anomaly_types)
                severity = np.random.uniform(0.5, 1.5)  # Varying severity
                
                effects = self._generate_anomaly_sequence(anomaly_type, duration, severity)
                
                # Apply effects
                for vital, effect in effects.items():
                    if vital in data:
                        data[vital][i:i+duration] += effect
                
                # Mark anomaly labels
                for j in range(i, i + duration):
                    data["is_anomaly"][j] = 1
                    data["anomaly_type"][j] = anomaly_type.name
                
                self.anomaly_labels[patient_id].append((i, i + duration, anomaly_type))
                
                # Skip past anomaly to avoid overlapping
                i += duration + np.random.randint(60, 180)  # Gap between anomalies
            else:
                i += 1
        
        return data
    
    def simulate_all_patients(
        self,
        num_patients: Optional[int] = None,
        duration_hours: Optional[float] = None,
        inject_anomalies: bool = True
    ) -> pd.DataFrame:
        """
        Simulate vital signs for all patients.
        
        Returns combined DataFrame for all patients.
        """
        num_patients = num_patients or self.sim_config.num_patients
        
        all_data = []
        for pid in range(num_patients):
            patient_data = self.simulate_patient(pid, duration_hours, inject_anomalies)
            all_data.append(patient_data)
            
            if self.config.debug and pid % 10 == 0:
                print(f"Simulated patient {pid}/{num_patients}")
        
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics of the generated dataset."""
        vitals = ["heart_rate", "spo2", "sbp", "dbp", "temperature", "resp_rate"]
        
        stats = {
            "total_samples": len(df),
            "num_patients": df["patient_id"].nunique(),
            "anomaly_rate": df["is_anomaly"].mean(),
            "anomaly_distribution": df[df["is_anomaly"] == 1]["anomaly_type"].value_counts().to_dict(),
            "vital_statistics": {}
        }
        
        for vital in vitals:
            stats["vital_statistics"][vital] = {
                "mean": df[vital].mean(),
                "std": df[vital].std(),
                "min": df[vital].min(),
                "max": df[vital].max()
            }
        
        return stats


def generate_dataset(
    num_patients: int = 100,
    duration_hours: float = 24,
    output_path: Optional[str] = None,
    debug: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to generate a complete dataset.
    
    Returns:
        - DataFrame with all patient data
        - Statistics dictionary
    """
    config = get_config()
    config.debug = debug
    
    simulator = IoTSensorSimulator(config)
    df = simulator.simulate_all_patients(num_patients, duration_hours)
    stats = simulator.get_statistics(df)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
    
    return df, stats


if __name__ == "__main__":
    # Generate sample dataset
    print("Generating simulated IoT sensor dataset...")
    df, stats = generate_dataset(
        num_patients=10,
        duration_hours=4,
        debug=True
    )
    
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Anomaly rate: {stats['anomaly_rate']*100:.2f}%")
    print(f"\nVital Sign Statistics:")
    for vital, vstats in stats["vital_statistics"].items():
        print(f"  {vital}: {vstats['mean']:.2f} ± {vstats['std']:.2f}")
