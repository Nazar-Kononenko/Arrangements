import torch

# MERT config
mert_model = "m-a-p/mert-v1-95M"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters for collecting transitions
beat_similarity_threshold = 0.8
energy_similarity_threshold = 0.8
min_duration = 2.0
max_duration = 30.0

# Parameters for arrangement scoring
duration_score_weight = 0.6
transition_score_weight = 0.1
segment_bonus_weight = 0.3

# Parameters for arrangement search
max_loops = 3
tolerance = 2.0
arrangement_amount = 10
