# Arrangements module documentation

## Collecting audio metadata

In order to collect audio metadata, you have to perform audio segmentation with `collect_metadata.py` module.

Arguments:
- `audio_file`: Path to the input audio file.

After running the script, the new folder will be created, named after the audio file, containing the following files:
- `metadata.json`: Contains the metadata of the audio file.
- `<copy of audio file>`: A copy of the original audio file.

Metadata is stored in `metadata.json` in the following format:
```json
{
  "audio_file": "path/to/audio/file",
  "duration": 30.0,
  "bpm": 120,
  "beats": [0.29,0.72,1.13,1.56,1.98,2.39],
  "downbeats": [0.29,1.13,1.98],
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 10.0,
      "label": "Segment 1"
    },
    {
      "start_time": 10.0,
      "end_time": 20.0,
      "label": "Segment 2"
    }
  ]
}
```

## Collecting beat transitions

Next step is to collect beat transitions using `collect_transitions.py` module. 

Arguments:
- `audio_folder`: Path to the folder containing the audio file and metadata (created in the previous step).

After running the script, the following files will be created in the same folder:
- `transitions.json`: Contains valid beat transitions of the audio.

These transitions should be used for creating required arrangements of your audio file.

## Creating arrangements

To create arrangements, you can use the `collect_arrangements.py` module.

Arguments:
- `audio_folder`: Path to the folder containing the audio file and metadata and set of transitions.
- `--target_duration`: Target duration of the arrangement in seconds (default is 30 seconds).
- `--verbose`: If set, will print additional information during the arrangement creation process.

After running the script, the following files will be created in the same folder:
- `arrangement.json`: Contains a list of arrangements of target length, obtained from the audio.

We return the amount of arrangements specified in `config.py` file. You can also see a lot of different parameters to tune there,

Example of arrangement:

```json
{
    "index": 1,
    "transitions": [
      {
        "type": "skip",
        "from_beat": 4,
        "to_beat": 56,
        "duration": 21.98
      },
      {
        "type": "skip",
        "from_beat": 116,
        "to_beat": 136,
        "duration": 9.31
      }
    ],
    "arrangement_duration": 30.71,
    "original_duration": 62.0,
    "target_duration": 30.0,
    "score": 0.6572285714285714
  }
```

## Using arrangements to rearrange audio

To rearrange audio using the created arrangements, you can use the `rearrange_audio.py` module.

Arguments:
- `audio_folder`: Path to the folder containing the audio file, metadata, transitions, and arrangements.
- `arrangement_index`: Index of the arrangement to use (default is 1, which is the first arrangement).

After running the script, newly arranged audio will be saved in `audio_folder/arrangement_<index>.wav`, where `<index>` is the index of the arrangement used.
