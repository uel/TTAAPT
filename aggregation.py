from miditoolkit.midi import parser as mid_parser
from miditoolkit.pianoroll import parser as pr_parser
import numpy as np
from sklearn import metrics

def get_pr(midi_file, fps=100):
    midi = mid_parser.MidiFile(midi_file)
    mapping = midi.get_tick_to_time_mapping()
    notes = midi.instruments[0].notes
    tick = notes[-1].end
    sec = mapping[tick]
    ticks_per_sec = tick / sec
    resample_factor = fps/ticks_per_sec
    return pr_parser.notes2pianoroll(midi.instruments[0].notes, (21, 108), resample_factor=resample_factor), resample_factor 

def aggregate_midi(in_paths, out_path, fps=100, mode="median", threshold=0.75): # 10ms hop size
    piano_rolls = []
    piano_rolls_bool = []
    for i, in_path in enumerate(in_paths):
        pr, resample_factor = get_pr(in_path, fps)
        piano_rolls.append(pr)
        piano_rolls_bool.append(pr > 0)
    
    max_length = min([pr.shape[0] for pr in piano_rolls])
    for i in range(len(piano_rolls)):
        piano_rolls_bool[i] = piano_rolls_bool[i][:max_length]
        piano_rolls[i] = piano_rolls[i][:max_length]

    if "threshold" in mode:
        threshold = float(mode.split("_")[1])
        aggregated_mask = np.mean(piano_rolls_bool, axis=0) > threshold
    elif mode == "median":
        aggregated_mask = np.median(piano_rolls_bool, axis=0)

    # aggregated_pr = np.mean(piano_rolls, axis=0) # mean used for velocity
    # aggregated_pr = aggregated_pr * aggregated_mask

    aggregated_pr = 100 * aggregated_mask 

    notes = pr_parser.pianoroll2notes(aggregated_pr, 1/resample_factor, (21, 108))
    midi = mid_parser.MidiFile(in_paths[0])
    midi.instruments[0].notes = notes

    midi.dump(out_path)

def calculate_f1_score(true_midi_file, pred_midi_file, fps=100):
    pr1, _ = get_pr(true_midi_file, fps)
    pr2, _ = get_pr(pred_midi_file, fps)

    min_len = min(pr1.shape[0], pr2.shape[0])
    pr1 = pr1[:min_len]
    pr2 = pr2[:min_len]

    pr1 = pr1 > 0
    pr2 = pr2 > 0
    
    f1 = metrics.f1_score(pr1.flatten(), pr2.flatten(), average='macro')
    
    return f1