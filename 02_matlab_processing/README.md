# MATLAB Data Processing

`make_fit_density_aligned_fixed.m` is the only retained MATLAB entry point. It aligns RFID observations with PT100 temperatures, filters outliers, and selects representative temperature-persistence-time points.

Its default file pickers resolve the versioned inputs under `../03_python_reproduction/data/`; newly collected RFID files can be selected from `../01_data_acquisition/output/`. Outputs are created under `02_matlab_processing/output/aligned/` and are ignored by Git.

Legacy timestamp-correction and exploratory plotting scripts are deliberately excluded because they are not required by the current paper pipeline.
