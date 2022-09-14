# https://ideone.com/tR0jFc

import torch
import glob
from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct
import numpy as np
import matplotlib.pyplot as plt


NOTE_SEQUENCE = [50, 52, 55, 57, 58, 59, 60, 62, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 76, 77, 79, 81]
NOTE_CHAR_SEQUENCE = ["D,", "E,", "G,", "A,", "_B,", "B,", "C", "D", "E", "F", "_G", "G","A", "_B", "B", "c", "_d", "d", "e", "f", "g", "a"]
NOTE_SEQUENCE = list(map(lambda pitch: f"n{pitch}", NOTE_SEQUENCE))
VOCABULARY = ["p", "s", "e"] +  NOTE_SEQUENCE
LEN_VOCAB = len(VOCABULARY)


def load_midi_folder(path: str) -> list:
    list_path = glob.glob(path)
    midi_list = [mid_parser.MidiFile(_path) for _path in list_path]
    print(f"[INFO] Load successfully {len(midi_list)} file midi from source")
    list_path = list(map(lambda _path: _path.split("\\")[-1], list_path))
    return midi_list, list_path


def midi2seq(midi_list: list) -> list:
    sequences = []
    tmp_cache = {
        "note": 0,
        "track": 0
    }
    for midi in midi_list:
        notes = midi.instruments[0].notes
        tmp_cache["track"] += (notes[-1].end//(2*midi.ticks_per_beat) + (notes[-1].end%(2*midi.ticks_per_beat) != 0))
        
        sub_pitch = 0
        if len(midi.key_signature_changes):
            sub_pitch = midi.key_signature_changes[0].key_number

        sequence = list(map(lambda note: note.pitch-sub_pitch, notes))
        
        tmp_cache["note"] += len(notes)
        sequences.append(sequence)

    print("[INFO] Convert midi to sequences successfully")
    print(f"\t[+] Note: {tmp_cache['note']}")
    print(f"\t[+] Track: {tmp_cache['track']}")
    
    return sequences


def seq2vocab_idx(sequences: list) -> list:
    token_idx = []
    for seq in sequences:
        seq_idx = list(map(lambda pitch: VOCABULARY.index(f"n{pitch}"), seq))
        token_idx.append(seq_idx)
    return token_idx


def vocab_idx2tensor(tokens_idx: list) -> torch.Tensor:
    tensor_data = torch.Tensor()
    max_length = 0
    for idx, seq_idx in enumerate(tokens_idx):
        seq_idx = torch.tensor(seq_idx)
        tokens_idx[idx] = seq_idx.view(1, len(seq_idx))
        max_length = max(max_length, len(seq_idx))
    
    for seq_idx in tokens_idx:
        pad_zeros = torch.zeros((1, max_length - seq_idx.size(1)))
        st_token = torch.tensor([[1]])
        en_token = torch.tensor([[2]])
        seq_idx = torch.cat([st_token, seq_idx, pad_zeros, en_token], dim=1)
        tensor_data = torch.cat((tensor_data, seq_idx), dim=0)
    return tensor_data


def plot_midi_roll(midi: mid_parser.MidiFile) -> None:
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.set_facecolor('White')
    start = np.zeros(len(NOTE_SEQUENCE))
    label = np.array(NOTE_SEQUENCE)
    
    notes = midi.instruments[0].notes
    for idx, note in enumerate(notes):
        pitch = note.pitch
        duration = (note.end - note.start)//30
        if idx == 0:
            start += (note.start//30)
        else:
            start += (notes[idx].start - notes[idx-1].end)//30
        width = np.zeros(len(NOTE_SEQUENCE))
        width[NOTE_SEQUENCE.index(f"n{pitch}")] = duration
        ax.barh(label, width, left=start, height=0.4, label=pitch, color="green")
        start += duration
        
    plt.xlabel("Time by tick")
    plt.ylabel("Pitch")
    plt.show()


if __name__ == "__main__":
    path = "./datasets/midi_songs/*.mid"
    midi_list, list_path = load_midi_folder(path)
    sequences = midi2seq(midi_list)
    tokens_idx = seq2vocab_idx(sequences)
    tensor_data = vocab_idx2tensor(tokens_idx)
    plot_midi_roll(midi_list[0])
    torch.save(tensor_data, "./data_gen/data_v7.pt")