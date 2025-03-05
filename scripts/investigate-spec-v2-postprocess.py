import pandas as pd

# read in /data/peer/ibrahih3/spectra/spec-v2-log/output_postprocessing/cui23_processed_spectrav2_example_20241216_spectra_epochs_10000-time_1736835731.1690764markers.csv
markers = pd.read_csv('/data/peer/ibrahih3/spectra/spec-v2-log/output_postprocessing/cui23_processed_spectrav2_example_20241216_spectra_epochs_10000-time_1736835731.1690764markers.csv')

il2 = markers[markers['geneset_match'] == 'Treg_IL2']

ifng = markers[markers['geneset_match'] == 'Treg_IFNG']

th1 = markers[markers['geneset_match'] == 'Treg_Th1']

# group by cell_type and sum over the importance column
il2.groupby('cell_type')['importance'].mean()

ifng.groupby('cell_type')['importance'].mean()

th1.groupby('cell_type')['importance'].mean()