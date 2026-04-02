"""
Medical Dataset
===============
Dataset of (diagnosis, prescription) pairs for training MedLLaMA.

Data format (JSON):
[
  {
    "diagnosis": "Patient presents with fever 102°F, productive cough, chest pain. 
                  Chest X-ray shows right lower lobe consolidation.",
    "prescription": "[DRUG] Amoxicillin [DOSE] 500mg [FREQ] three times daily for 7 days
                     [DRUG] Paracetamol [DOSE] 500mg [FREQ] as needed for fever
                     [WARN] Rest and adequate hydration. Follow up in 5 days."
  },
  ...
]

The model learns to generate the prescription given the diagnosis.
Training format:
  [DIAG] <diagnosis text> [SEP] [PRESC] <prescription text> [END]
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import MedTokenizer, SPECIAL_TOKENS


# ─────────────────────────────────────────────────────────────────────────────
# Sample medical data (used when no real dataset is provided)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_DATA = [
    {
        "diagnosis": "Patient: 35F. Fever 101°F for 3 days, productive cough with yellowish sputum, chest pain on inspiration. SpO2 96%. Chest X-ray: right lower lobe consolidation. WBC 14,000. Diagnosis: Community-acquired pneumonia.",
        "prescription": "[DRUG] Amoxicillin-Clavulanate [DOSE] 625mg [FREQ] twice daily for 7 days [DRUG] Azithromycin [DOSE] 500mg [FREQ] once daily for 5 days [DRUG] Paracetamol [DOSE] 650mg [FREQ] every 6 hours as needed for fever [DRUG] Mucolytic (Ambroxol) [DOSE] 30mg [FREQ] three times daily [WARN] Adequate rest and oral hydration. Return if SpO2 drops below 93% or no improvement in 48 hours."
    },
    {
        "diagnosis": "Patient: 52M. Chief complaint: crushing chest pain radiating to left arm, diaphoresis, nausea for 2 hours. BP 150/95, HR 102. ECG: ST elevation in leads II, III, aVF. Troponin elevated. Diagnosis: Acute Inferior STEMI.",
        "prescription": "[DRUG] Aspirin [DOSE] 325mg [FREQ] immediately (loading dose), then 75mg once daily [DRUG] Clopidogrel [DOSE] 600mg [FREQ] immediately (loading dose), then 75mg once daily [DRUG] Heparin [DOSE] 5000 IU IV [FREQ] stat, then infusion [DRUG] Nitroglycerin [DOSE] 0.4mg sublingual [FREQ] every 5 minutes x3 if pain persists [DRUG] Morphine [DOSE] 2-4mg IV [FREQ] for pain relief [WARN] Immediate PCI referral. Monitor vitals continuously. Avoid NSAIDs."
    },
    {
        "diagnosis": "Patient: 28F. Dysuria, urinary frequency, urgency for 2 days. No fever. Urine dipstick: nitrites positive, leukocytes 3+. No costovertebral angle tenderness. Diagnosis: Uncomplicated urinary tract infection (cystitis).",
        "prescription": "[DRUG] Nitrofurantoin [DOSE] 100mg modified-release [FREQ] twice daily for 5 days [DRUG] Phenazopyridine [DOSE] 200mg [FREQ] three times daily for 2 days (symptom relief) [WARN] Drink 2-3L water daily. Avoid caffeine and alcohol. Urine culture if no improvement in 48 hours. Avoid nitrofurantoin if creatinine clearance < 45 mL/min."
    },
    {
        "diagnosis": "Patient: 45M. Fasting blood glucose 210 mg/dL, HbA1c 9.2%, polyuria, polydipsia, weight loss 8kg over 3 months. BMI 28. No ketones in urine. BP 138/88. Diagnosis: Type 2 Diabetes Mellitus, newly diagnosed.",
        "prescription": "[DRUG] Metformin [DOSE] 500mg [FREQ] twice daily with meals, increase to 1000mg after 2 weeks if tolerated [DRUG] Linagliptin [DOSE] 5mg [FREQ] once daily [DRUG] Amlodipine [DOSE] 5mg [FREQ] once daily for hypertension [WARN] Low carbohydrate diet. 30 min exercise daily. Blood glucose monitoring 2x/day. Review in 3 months with HbA1c. Watch for metformin GI side effects."
    },
    {
        "diagnosis": "Patient: 8M (child). Fever 103°F, sore throat, difficulty swallowing, exudates on tonsils, tender anterior cervical lymph nodes. Rapid strep test positive. No cough. Diagnosis: Group A Streptococcal tonsillitis.",
        "prescription": "[DRUG] Amoxicillin [DOSE] 50mg/kg/day divided twice daily [FREQ] for 10 days (max 1g/day) [DRUG] Paracetamol [DOSE] 15mg/kg [FREQ] every 4-6 hours as needed (max 5 doses/day) [DRUG] Chlorhexidine gargle [DOSE] 10ml [FREQ] twice daily (if child can gargle) [WARN] Complete full antibiotic course. Soft diet. Plenty of fluids. Return if rash develops (possible scarlet fever). Avoid school for 24 hours after starting antibiotics."
    },
    {
        "diagnosis": "Patient: 62F. Progressively worsening knee pain, worse with activity, morning stiffness < 30 minutes, crepitus on movement. X-ray: joint space narrowing, osteophytes, subchondral sclerosis. BMI 32. Diagnosis: Knee osteoarthritis, moderate.",
        "prescription": "[DRUG] Paracetamol [DOSE] 500mg [FREQ] three times daily (regular, not PRN) [DRUG] Diclofenac gel 1% [DOSE] 4g [FREQ] apply to affected knee four times daily [DRUG] Glucosamine + Chondroitin [DOSE] 1500mg + 1200mg [FREQ] once daily [WARN] Physiotherapy referral mandatory. Weight reduction target BMI < 27. Avoid prolonged stair climbing. Use knee support. Avoid oral NSAIDs if GI history. Intra-articular corticosteroid injection if inadequate relief."
    },
    {
        "diagnosis": "Patient: 33M. Generalized anxiety, palpitations, sweating, tremors for 6 months. Weight loss 5kg. TSH < 0.01, Free T4 elevated. Thyroid scan: diffuse uptake. Exophthalmos present. Diagnosis: Graves' disease (hyperthyroidism).",
        "prescription": "[DRUG] Carbimazole [DOSE] 20mg [FREQ] three times daily initially, taper after 4-6 weeks [DRUG] Propranolol [DOSE] 40mg [FREQ] twice daily for symptom control (tremor, palpitations) [DRUG] Artificial tears [DOSE] 1-2 drops [FREQ] four times daily for eye symptoms [WARN] Monitor TFTs every 4-6 weeks. Watch for agranulocytosis (sore throat → stop drug, emergency CBC). Endocrinology referral. Ophthalmology for eye disease. Radioiodine or surgery may be needed."
    },
    {
        "diagnosis": "Patient: 19F. Severe unilateral throbbing headache, photophobia, phonophobia, nausea, vomiting. Duration 8 hours. Prior episodes similar. No fever, neck stiffness, or neurological deficit. Normal neurological exam. Diagnosis: Migraine without aura, acute attack.",
        "prescription": "[DRUG] Sumatriptan [DOSE] 50mg oral [FREQ] at onset, repeat 100mg after 2 hours if partial response (max 300mg/24h) [DRUG] Metoclopramide [DOSE] 10mg [FREQ] for nausea/vomiting [DRUG] Ibuprofen [DOSE] 400mg [FREQ] with triptan if triptan alone insufficient [WARN] Rest in dark quiet room. Ice pack on forehead. Maintain headache diary. If > 4 attacks/month refer to neurology for prophylaxis (propranolol/topiramate). Avoid ergotamines + triptans simultaneously."
    },
    {
        "diagnosis": "Patient: 55M. Smoker 30 pack-years. Productive cough >3 months for 2 consecutive years, dyspnea on exertion, wheeze. FEV1/FVC < 70%, FEV1 55% predicted post-bronchodilator. SpO2 94%. Diagnosis: COPD, GOLD Stage II (moderate).",
        "prescription": "[DRUG] Tiotropium [DOSE] 18mcg [FREQ] once daily via HandiHaler (LAMA) [DRUG] Salmeterol/Fluticasone [DOSE] 50/250mcg [FREQ] twice daily via inhaler (LABA+ICS) [DRUG] Salbutamol [DOSE] 100mcg [FREQ] 2 puffs as needed (rescue) [DRUG] Carbocisteine [DOSE] 375mg [FREQ] three times daily [WARN] Smoking cessation — MANDATORY and most important intervention. Pulmonary rehabilitation. Annual influenza vaccine. Pneumococcal vaccine. Check inhaler technique. Refer to pulmonology."
    },
    {
        "diagnosis": "Patient: 40F. Sudden painless loss of vision in left eye, pale retina with cherry red spot at fovea on fundoscopy. BP 160/100. History of atrial fibrillation. Duration < 2 hours. Diagnosis: Central Retinal Artery Occlusion (CRAO) — ophthalmologic emergency.",
        "prescription": "[DRUG] Acetazolamide [DOSE] 500mg IV [FREQ] immediately [DRUG] Aspirin [DOSE] 300mg [FREQ] immediately [WARN] EMERGENCY — immediate ophthalmology referral within minutes. Digital massage of the globe. Anterior chamber paracentesis if available. Hyperventilation into paper bag (increase CO2). Stroke unit admission — high risk of concurrent ischaemic stroke. Anticoagulation after embolic workup. Carotid duplex and cardiac echo urgently."
    },
    {
        "diagnosis": "Patient: 25F. Butterfly-shaped rash across cheeks and nose, joint pain, fatigue, oral ulcers, photosensitivity. ANA positive 1:640, anti-dsDNA positive, low complement (C3, C4). Urine protein 2+. Diagnosis: Systemic Lupus Erythematosus (SLE).",
        "prescription": "[DRUG] Hydroxychloroquine [DOSE] 200mg [FREQ] twice daily (disease-modifying, reduces flares) [DRUG] Prednisolone [DOSE] 30mg [FREQ] once daily, taper over 6-8 weeks [DRUG] Sunscreen SPF 50+ [FREQ] apply daily [DRUG] Calcium + Vitamin D [DOSE] 1000mg + 800IU [FREQ] once daily (steroid bone protection) [WARN] Rheumatology and nephrology referral. Avoid sun exposure. Monitor CBC, LFTs, ophthalmology for hydroxychloroquine retinopathy annually. Avoid live vaccines."
    },
    {
        "diagnosis": "Patient: 3M infant. Fever 39.5°C for 24h, bulging fontanelle, neck stiffness, purpuric rash, photophobia, irritability, high-pitched cry. CSF: turbid, elevated WBC (neutrophils), low glucose, high protein. Gram-negative diplococci. Diagnosis: Bacterial meningitis (Neisseria meningitidis).",
        "prescription": "[DRUG] Ceftriaxone [DOSE] 100mg/kg/day IV divided q12h [FREQ] for 7 days [DRUG] Dexamethasone [DOSE] 0.15mg/kg IV [FREQ] every 6 hours for 4 days (give before first antibiotic) [WARN] MEDICAL EMERGENCY — do not delay antibiotics. Isolate patient. Public health notification mandatory. Close contacts: Rifampicin prophylaxis. Intensive care monitoring. Monitor for complications: hearing loss, hydrocephalus, DIC."
    },
]


def create_sample_dataset(path: str = "data/medical_data.json"):
    """Save sample data to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(SAMPLE_DATA, f, indent=2)
    print(f"Sample dataset created: {path} ({len(SAMPLE_DATA)} examples)")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MedicalDataset(Dataset):
    """
    PyTorch Dataset for diagnosis → prescription pairs.
    
    Each training example is formatted as:
      [DIAG] {diagnosis} [SEP] [PRESC] {prescription} [END]
    
    The model is trained with teacher forcing:
      input  = full sequence[:-1]
      target = full sequence[1:]  (next-token prediction)
    """

    def __init__(self, data: List[Dict], tokenizer: MedTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.examples  = []

        for item in data:
            diag  = item["diagnosis"].strip()
            presc = item["prescription"].strip()
            # Format: [DIAG] diagnosis [SEP] [PRESC] prescription [END]
            text = f"[DIAG] {diag} [SEP] [PRESC] {presc} [END]"
            ids  = tokenizer.encode(text, add_bos=True, add_eos=True, max_len=max_len)
            if len(ids) > 2:
                self.examples.append(ids)

        print(f"Dataset: {len(self.examples)} examples loaded")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.examples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)   # input: all but last
        y = torch.tensor(ids[1:],  dtype=torch.long)   # target: all but first
        return x, y


def collate_fn(batch, pad_id: int = 0):
    """Pad batch to same length."""
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)

    x_pad = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    y_pad = torch.full((len(ys), max_len), pad_id, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :x.size(0)] = x
        y_pad[i, :y.size(0)] = y

    return x_pad, y_pad


def get_dataloaders(data_path: str, tokenizer: MedTokenizer,
                    max_len: int = 512, batch_size: int = 8,
                    val_split: float = 0.1) -> Tuple[DataLoader, DataLoader]:
    """Load data, split into train/val, return DataLoaders."""
    with open(data_path) as f:
        data = json.load(f)

    random.shuffle(data)
    split = max(1, int(len(data) * (1 - val_split)))
    train_data = data[:split]
    val_data   = data[split:] or data[:1]  # at least 1 val example

    train_ds = MedicalDataset(train_data, tokenizer, max_len)
    val_ds   = MedicalDataset(val_data,   tokenizer, max_len)

    pad_id = tokenizer.special_tokens["<pad>"]
    _collate = lambda b: collate_fn(b, pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=_collate)

    return train_loader, val_loader
