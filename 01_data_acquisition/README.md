# RFID Data Acquisition

Only `src/main/java/org/example/MultiTagHeatCollector.java` is retained. The exploratory Java files from the original IDE project are not needed for reproducing the paper workflow.

Obtain `OctaneSDKJava-1.26.0.0-jar-with-dependencies.jar` from Impinj and place it under `lib/`, then configure the constants at the top of `MultiTagHeatCollector.java`.

The default destination is `01_data_acquisition/output/<timestamp>/`. Set `RFID_ACQUISITION_OUTPUT` to use another data disk. The folder is created at runtime and ignored by Git.
