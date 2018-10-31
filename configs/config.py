# ------------ Config ---------------
class NasdaqConfig:
    input_keep_prob = 1.0
    output_keep_prob = 1.0
    
    input_size = 1
    output_size = 1

    batch_size = 128
    
    hidden_sizes = [64]

    n = 81
    
    timestep = 10

    lr = 0.001

    is_scaled = True

    feature_range = (0, 10)

class TaxiNYConfig:
    input_keep_prob = 0.8
    output_keep_prob = 1.0
    
    input_size = 1
    output_size = 1

    batch_size = 128
    
    hidden_sizes = [64, 32]

    n = 99
    
    timestep = 12

    lr = 0.001

    is_scaled = True

    feature_range = (0, 1)
