import aggregation
import augmentation
import piano_transcription_inference
import librosa

def create_files():
    # create the augmented audio files
    for pitch_shift in [-3, -2, -1, 0, 1, 2, 3]:
      augmentation.augment_audio('samples/1.mp3', f'augmented/1_shift_{pitch_shift}.wav', 1, pitch_shift)

    # transcribe the augmented audio
    pt = piano_transcription_inference.PianoTranscription(checkpoint_path="C:/Users/danif/s/BP/models/piano_transcription.pth", device='cpu')
    for pitch_shift in [-3, -2, -1, 0, 1, 2, 3]:
        audio = librosa.load(f'augmented/1_shift_{pitch_shift}.wav', sr=16000)[0]
        pt.transcribe(audio, f'transcribed/1_shift_{pitch_shift}.mid')
    
    # reverse the pitch shift on the transcribed midi files
    for pitch_shift in [-3, -2, -1, 0, 1, 2, 3]:
        augmentation.augment_midi(f'transcribed/1_shift_{pitch_shift}.mid', f'reversed/1_shift_{pitch_shift}.mid', 1, -pitch_shift)

def evaluate_augmented():
    # calculate the f1 score
    for pitch_shift in range(-3, 4):
        f1 = aggregation.calculate_f1_score('samples/1.mid', f'reversed/1_shift_{pitch_shift}.mid', 100)
        if pitch_shift != 0:
            print(f'Pitch shift {pitch_shift} F1: {f1}')
        else:
            print(f'Baseline F1: {f1}')
    print()

def evaluate_aggregate(mode="median", spread=3):
    pitch_range = range(-spread, spread+1)

    # aggregate the transcribed midi files
    transcribed_midi_files = [f'reversed/1_shift_{pitch_shift}.mid' for pitch_shift in pitch_range]
    aggregation.aggregate_midi(transcribed_midi_files, 'aggregated.mid', 100, mode)

    f1 = aggregation.calculate_f1_score('samples/1.mid', 'aggregated.mid', 100)
    print(f'{-spread} to {spread} shifted {mode} aggregated F1: {f1}')


if __name__ == '__main__':
    # create_files()
    evaluate_augmented()

    for pitch_range in [1, 2, 3]:
        evaluate_aggregate("median", pitch_range)
        evaluate_aggregate("threshold_0.25", pitch_range)
        evaluate_aggregate("threshold_0.75", pitch_range)
        print()