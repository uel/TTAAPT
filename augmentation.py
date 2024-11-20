import os
import mido

def augment_audio(in_path, out_path, stretch_multiplier, semitones):
    os.system(f'rubberband -t {stretch_multiplier} -p {semitones} {in_path} {out_path}')

def augment_midi(in_path, out_path, stretch_multiplier, semitones):
    midi = mido.MidiFile(in_path)
    for track in midi.tracks:
        for msg in track:
            msg.time = int(msg.time * stretch_multiplier) # check if matches audio, pretty midi stereo
            if msg.type == 'note_on' or msg.type == 'note_off':
                msg.note += semitones
    midi.save(out_path)
