# Medical or General-Purpose LLMs for Virtual Patient Simulation?

**A Comparative Evaluation in OSCE-Aligned Scenarios**

Published at [IEEE SeGAH 2026](https://www.segah.org/2026/) — IEEE International Conference on Serious Games and Applications for Health

---

## About

This repository accompanies our SeGAH 2026 paper, which presents the first systematic comparison of medical-specialized and general-purpose LLMs for virtual patient simulation in medical education.

We evaluated **8 open-source LLMs** (4 medical, 4 general-purpose) across **8 OSCE-aligned clinical scenarios** at 3 difficulty levels with 3 patient personas each, producing **576 unique simulated conversations**. Each conversation was scored using a dual evaluation framework combining 7 practical deployment metrics with 6 literature-derived metrics grounded in the RIAS communication coding system and the CARE empathy scale, and validated by three blinded clinicians.

Our results show that base model architecture and fine-tuning quality are stronger predictors of simulation quality than whether a model carries a "medical" label. General-purpose models outperformed medical models at the group level, but the top four individual models — a mix of both categories — were statistically indistinguishable, all scoring above 4.6/5 on human evaluation.

---

## Dataset

`LLM_Conversations.json` contains all 576 conversation transcripts. Each entry includes:

| Field | Description |
|---|---|
| `model` | LLM used (e.g., `medqa-llama3-8b`, `llama-3.1-8b-instruct`) |
| `category` | `medical` or `general` |
| `scenario_id` | Clinical scenario (e.g., `chest_pain_acs`, `depression`) |
| `difficulty` | `easy`, `moderate`, or `hard` |
| `patient_name` | Randomly generated patient persona name |
| `messages` | Array of `{role, content}` pairs (10 examiner questions + 10 patient responses) |
| `response_times` | Per-turn inference latency in seconds |
| `total_time` | Total conversation generation time |

### Models

| Model | Category | Base Architecture | Parameters |
|---|---|---|---|
| MedQA-Llama3-8B | Medical | LLaMA-3.1 | 8B |
| Bio-Medical-Llama3-8B | Medical | LLaMA-3 | 8B |
| Medical-Llama3-8B | Medical | LLaMA-3 | 8B |
| OpenBioLLM-8B | Medical | LLaMA-3 | 8B |
| Qwen2.5-7B-Instruct | General | Qwen2.5 | 7B |
| LLaMA-3.1-8B-Instruct | General | LLaMA-3.1 | 8B |
| Mistral-7B-Instruct-v0.3 | General | Mistral | 7B |
| Mistral-Nemo-12B-Instruct | General | Mistral | 12B |

### Clinical Scenarios

Chest Pain (ACS), Community-Acquired Pneumonia, Type 2 Diabetes, Anaphylactic Reaction, Acute Appendicitis, Major Depression, Acute Confusion (Geriatric), and Pediatric Fever.

---

## Citation

```bibtex
@inproceedings{aloraini2026vp,
  title={Medical or General-Purpose LLMs for Virtual Patient Simulation? A Comparative Evaluation in OSCE-Aligned Scenarios},
  author={Aloraini, Abdulrahman},
  booktitle={Proceedings of the IEEE International Conference on Serious Games and Applications for Health (SeGAH)},
  year={2026},
  organization={IEEE}
}
```

---

## License

This dataset is released for research purposes. If you use this data, please cite the paper above.

## Contact

Abdulrahman Aloraini — Department of Information Technology, College of Computing, Qassim University  
a.aloraini@qu.edu.sa
