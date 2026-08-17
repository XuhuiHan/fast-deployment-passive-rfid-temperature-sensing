# Versioned experimental inputs

This directory contains only the input measurements required to rerun the paper workflow:

- `training/studydata.txt`: learning-tag temperature and persistence-time observations;
- `validation/tags/`: RFID response records for the sliding-window and one-point evaluations;
- `temperature/123time_time_corrected.csv`: reference-temperature record.

Pretrained model JSON files and expected-result summaries are intentionally absent. They are generated under `../outputs/` when the pipeline runs.
