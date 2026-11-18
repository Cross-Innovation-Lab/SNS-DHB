# submission files

`sorted/` contains the WaveClus/Phy sorting output for Participant 1 (selected cluster 0) and Participant 2 (selected cluster 5). Spike times are sample indices at 30 kHz. 

`data/` contains the derived values required for the current figure: included trials, picture-aligned spikes, mean waveforms, unit quality summary, phase spike times, and single-spike waveform similarities.

`make_fig2.py` draws the current Figure 2. Run `python make_fig2.py` from this directory; the PDF and PNG are written to `figures/`. Tested with Python 3.11.15, NumPy 2.4.6, pandas 2.3.3, and Matplotlib 3.11.1. No NS6 or preprocessed voltage is included.
