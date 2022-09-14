import string
import torch
from model import Transformer
from tokenizer import load_midi_folder, VOCABULARY, NOTE_SEQUENCE, NOTE_CHAR_SEQUENCE
import random 
from fractions import Fraction


def load_model(filepath: str) -> Transformer:
    file = torch.load(filepath)
    model = Transformer(**file["hyper_params"]).to("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.load_state_dict(file["model_state_dict"])
    model.eval()
    return model.to("cuda:0" if torch.cuda.is_available() else "cpu")


def generate(model: Transformer, x: list) -> torch.Tensor:
    x = torch.tensor(x).to("cuda:0" if torch.cuda.is_available() else "cpu").long()
    x = x.unsqueeze(0)
    with torch.no_grad():
        while True:
            pred = model(x)
            k_top_pred = torch.topk(pred[0, -1, :], k=8, dim=0)
            pred_idx = torch.distributions.Categorical(logits=k_top_pred.values).sample()

            pred = k_top_pred.indices[pred_idx]
            if pred.item() == 2:
                break
            if x.size(1) > 363:
                break
            x = torch.cat((x, pred.view(1, 1)), dim=1)
    return x.squeeze()[1:]


def token2midi(tokens: torch.Tensor, save_path: str) -> None:
    path = "./datasets/midi_songs/*.mid"
    midi_list, list_path = load_midi_folder(path)
    midi_list = list(filter(lambda midi: len(midi.time_signature_changes) == 1 ,midi_list))
    midi_list = list(filter(lambda midi: midi.time_signature_changes[0].numerator==2, midi_list))
    midi_list = list(filter(lambda midi: midi.time_signature_changes[0].denominator==4, midi_list))
    N = len(midi_list)
    idx = random.randint(0, N-1)
    _notes = []
    midi_object = midi_list[idx]
    midi_notes = midi_object.instruments[0].notes
    min_length = min(len(midi_notes), len(tokens))
    for idx_note in range(min_length):
        current_note = midi_notes[idx_note]
        pitch = VOCABULARY[int(tokens[idx_note])].split("n")[-1]
        current_note.pitch = int(pitch)
        _notes.append(current_note)
    midi_object.instruments[0].notes = _notes 
    print("[INFO] Save successfully new midi file:", save_path)
    midi_object.dump(save_path)
    return midi_object


def midi2raw_data(midi: object) -> list:
    ticks = 2*midi.ticks_per_beat
    notes = midi.instruments[0].notes
    for i in range(1, len(notes)):
        if notes[i].start - notes[i].end < 60:
            notes[i-1].end = notes[i].start
    len_track = midi.max_tick//ticks
    raw_data = [[] for _ in range(len_track+1)]
    for idx, note in enumerate(notes):
        duration = note.end - note.start
        idx_pitch = NOTE_SEQUENCE.index(f"n{note.pitch}")
        note_char = NOTE_CHAR_SEQUENCE[idx_pitch]
        frac = Fraction(duration, ticks)
        track_id = note.start//ticks
        if frac.numerator == 1:
            value = frac.denominator
            raw_data[track_id].append(f"{note_char}{8//value}")
        else:
            if frac.numerator != 3:
                continue
            value = frac.denominator//2
            raw_data[track_id].append(f"{note_char}{8//value}>")
    string_data = []
    for data in raw_data:
        string_data.append(" ".join(data))
        if len(string_data[-1]) == 0:
            string_data = string_data[:-1]
    n_block = len(string_data)//3 + (len(string_data)%3 != 0)
    raw_data = []
    for i in range(n_block):
        st = i*3
        en = (i+1)*3
        raw_data.append(" | ".join(string_data[st: en]))
    return "\n".join(raw_data)


if __name__ == "__main__":
    model_note = load_model("./checkpoints/model_v7.pt")
    gen_data = generate(model_note, [1])
    midi_object= token2midi(gen_data,"./midi_gen/music1.mid")