import json
import random
import csv
from faker import Faker

fake = Faker()

form_factors = ["Portable scanner", "Rack server", "Embedded device", "Software only", "Cloud service", "Hybrid cloud"]
os_classes = ["Linux Ubuntu 22.04", "Linux Debian 12", "Linux CentOS 8", "Windows 10 Pro (64-bit)", 
              "Windows 11 Enterprise", "macOS Ventura", "Cloud-native Kubernetes", "Dockerized Linux",
              "Android 14", "iOS 17", "Embedded RTOS (VxWorks)", "QNX Neutrino RTOS", "Windows Server 2019"]
ui_classes = ["Web Dashboard", "Mobile App (iOS)", "Mobile App (Android)", "Desktop Client (Windows)",
              "Desktop Client (Linux)", "Touch Control Panel", "Voice Command Interface", 
              "Augmented Reality Viewer", "Virtual Reality Workspace", "3D Visualization Console"]

form_factor2id = {ff: i for i, ff in enumerate(form_factors)}
os2id = {os: i for i, os in enumerate(os_classes)}
ui2id = {ui: i for i, ui in enumerate(ui_classes)}



def generate_synthetic_device(idx):
    # Define logical pools
    logic = {
        "Cloud service": {
            "processors": ["NVIDIA A100", "NVIDIA H100", "Google TPU v4", "AWS Inferentia2", "Cloud GPU"],
            "os": ["Cloud-native Kubernetes", "Dockerized Linux", "Linux Ubuntu 22.04"],
            "uis": ["Web Dashboard"],
            "apis": ["FHIR API", "Custom REST API", "GraphQL API"]
        },
        "Hybrid cloud": {
            "processors": ["NVIDIA A100", "Google TPU v4", "Intel Xeon Gold 6330"],
            "os": ["Cloud-native Kubernetes", "Linux Ubuntu 22.04"],
            "uis": ["Web Dashboard", "Desktop Client (Linux)"],
            "apis": ["FHIR API", "Custom REST API"]
        },
        "Embedded device": {
            "processors": ["ARM Cortex-A72", "ARM Neoverse N2", "NVIDIA Jetson Orin"],
            "os": ["Embedded RTOS (VxWorks)", "QNX Neutrino RTOS", "Linux Debian 12"],
            "uis": ["Touch Control Panel", "Mobile App (Android)", "Voice Command Interface"],
            "apis": ["DICOM API"]
        },
        "Portable scanner": {
            "processors": ["NVIDIA Jetson Orin", "ARM Cortex-A72", "Apple M1"],
            "os": ["Linux Debian 12", "Android 14", "iOS 17"],
            "uis": ["Mobile App (iOS)", "Mobile App (Android)", "Touch Control Panel"],
            "apis": ["DICOM API", "FHIR API"]
        },
        "Rack server": {
            "processors": ["Intel Xeon Silver 4310", "Intel Xeon Gold 6330", "AMD EPYC 7763", "AMD EPYC 9654"],
            "os": ["Linux Ubuntu 22.04", "Windows Server 2019", "Windows 11 Enterprise"],
            "uis": ["Desktop Client (Windows)", "Web Dashboard"],
            "apis": ["FHIR API", "HL7 API"]
        },
        "Software only": {
            "processors": ["N/A", "Cloud GPU", "Intel Xeon Gold 6330"],
            "os": ["Linux Ubuntu 22.04", "Windows 10 Pro (64-bit)", "macOS Ventura"],
            "uis": ["Web Dashboard", "Desktop Client (Linux)"],
            "apis": ["FHIR API", "Custom REST API", "HL7 API"]
        }
    }

    # AI model-task mapping
    ai_logic = {
        "CNN": ["Image classification", "CT scan anomaly detection"],
        "UNet": ["MRI brain segmentation", "Organ boundary detection"],
        "ResNet-50": ["X-ray classification", "Feature extraction"],
        "YOLOv8": ["Lesion detection", "Tumor localization"],
        "Transformer": ["Multi-modal report analysis", "EHR classification"],
        "Vision Transformer (ViT)": ["Medical image classification", "Pathology slide analysis"],
        "Random Forest": ["Risk prediction", "Outcome classification"],
        "XGBoost": ["Tabular clinical prediction", "Biomarker risk scoring"],
        "Hybrid DL": ["Image + signal fusion", "Cross-modality analysis"]
    }

    # Pick form_factor logically
    form_factor = random.choice(list(logic.keys()))
    rules = logic[form_factor]
  
    model_type = random.choice(list(ai_logic.keys()))
    tasks = random.sample(ai_logic[model_type], k=min(2, len(ai_logic[model_type])))

    device = {
        "device_id": f"synthetic_{idx:04d}",
        "source": "synthetic",
        "device_name": fake.company() + " " + random.choice(["MRI Assist", "CT Vision", "UltraScan AI", "CardioTrack", "NeuroVision"]),
        "manufacturer": fake.company(),

        "hardware": {
            "processor": random.choice(rules["processors"]),
            "memory": random.choice(["8GB", "16GB", "32GB", "64GB", "128GB"]),
            "form_factor": form_factor
        },

        "software": {
            "os": random.choice(rules["os"]),
            "programming_language": random.choice(["Python", "C++", "Java", "Rust"]),
            "frameworks": random.sample(["TensorFlow", "PyTorch", "Keras", "ONNX Runtime"], random.randint(1, 2))
        },

        "ai_models": {
            "model_type": model_type,
            "tasks": tasks
        },

        "data_pipelines": {
            "input_format": ["DICOM"],
            "preprocessing": random.sample(["Normalization", "Resizing", "Noise Reduction", "Segmentation"], random.randint(1, 3)),
            "storage": random.choice(["Local", "Cloud PACS"]),
            "output_format": random.sample(["HL7", "FHIR", "JSON"], random.randint(1, 2))
        },

        "user_interface": {
            "type": random.choice(rules["uis"]),
            "features": random.sample(["Alert System", "Visualization", "3D Reconstruction", "Reporting"], random.randint(2, 3))
        },

        "integrations": {
            "hospital_systems": random.sample(["PACS", "RIS", "EMR", "EHR"], random.randint(1, 2)),
            "apis": random.sample(rules["apis"], random.randint(1, len(rules["apis"])))
        },

        "approval_year": random.randint(2015, 2025)
    }

    return device

def load_real_fda_devices(path="real_fda_devices.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("real_fda_devices.json not found. Using empty list.")
        return []
    
def create_hybrid_dataset(n_synthetic=50000, real_fda_path="real_fda_devices.json"):
    synthetic_devices = [generate_synthetic_device(i) for i in range(n_synthetic)]
    real_devices = load_real_fda_devices(real_fda_path)
    
    dataset = synthetic_devices + real_devices

    with open("hybrid_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "device_id", "source", "device_name", "manufacturer",
            "processor", "memory", "form_factor",
            "os", "programming_language", "frameworks",
            "model_type", "tasks",
            "input_format", "preprocessing", "storage", "output_format",
            "ui_type", "ui_features",
            "hospital_systems", "apis",
            "approval_year"
        ])
        
        for d in dataset:
            writer.writerow([
                d["device_id"],
                d["source"],
                d["device_name"],
                d["manufacturer"],
                d["hardware"]["processor"],
                d["hardware"]["memory"],
                d["hardware"]["form_factor"],
                d["software"]["os"],
                d["software"]["programming_language"],
                ";".join(d["software"]["frameworks"]),
                d["ai_models"]["model_type"],
                ";".join(d["ai_models"]["tasks"]),
                ";".join(d["data_pipelines"]["input_format"]),
                ";".join(d["data_pipelines"]["preprocessing"]),
                d["data_pipelines"]["storage"],
                ";".join(d["data_pipelines"]["output_format"]),
                d["user_interface"]["type"],
                ";".join(d["user_interface"]["features"]),
                ";".join(d["integrations"]["hospital_systems"]),
                ";".join(d["integrations"]["apis"]),
                d["approval_year"]
            ])
    
    print(f"Hybrid dataset created with {len(dataset)} devices (synthetic={len(synthetic_devices)}, real={len(real_devices)})")
    return dataset

def export_multitask_csv(json_path="hybrid_dataset.json", csv_path="multitask_dataset.csv"):
    with open(json_path, "r") as f:
        dataset = json.load(f)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "device_id", "source", "device_name", "manufacturer", "text_input",
            "form_factor", "form_factor_label",
            "os", "os_label",
            "ui_type", "ui_label"
        ])

        for d in dataset:
            ff = d["hardware"]["form_factor"]
            os = d["software"]["os"]
            ui = d["user_interface"]["type"]

            text_input = f"{d['hardware']['processor']} {d['hardware']['memory']} {ff} " \
                         f"{os} {d['software']['programming_language']} {' '.join(d['software']['frameworks'])} " \
                         f"{d['ai_models']['model_type']} {' '.join(d['ai_models']['tasks'])}"

            writer.writerow([
                d["device_id"], d["source"], d["device_name"], d["manufacturer"], text_input,
                ff, form_factor2id.get(ff, -1),
                os, os2id.get(os, -1),
                ui, ui2id.get(ui, -1)
            ])

    print(f"✅ Multi-task dataset saved to {csv_path}")

# ---------- Run ----------
if __name__ == "__main__":
    create_hybrid_dataset(n_synthetic=50000, real_fda_path="real_fda_devices.json")

    export_multitask_csv("multitask_dataset.csv")
