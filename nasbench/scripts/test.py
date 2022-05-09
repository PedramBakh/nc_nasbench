import sys
import os

# Add submodule to path
path = os.path.join(os.path.dirname(os.getcwd()), 'nc_carbontracker')
sys.path.append(path)

from carbontracker.tracker import CarbonTracker

tracker = CarbonTracker(epochs=1)

# Training loop.
for epoch in range(1):
    tracker.epoch_start()

    # Your model training.

    tracker.epoch_end()

# Optional: Add a stop in case of early termination before all monitor_epochs has
# been monitored to ensure that actual consumption is reported.
tracker.stop()


