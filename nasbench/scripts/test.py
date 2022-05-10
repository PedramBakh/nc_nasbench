import sys
import os
import json
import time

from nasbench.scripts.run_evaluation import NumpyEncoder

# Add submodule to path
path = os.path.join(os.path.dirname(os.getcwd()), 'nc_carbontracker')
sys.path.append(path)

from carbontracker.tracker import CarbonTracker
from carbontracker import parser

log_dir = os.path.join(os.getcwd(), 'nasbench', 'scripts', 'test_log')
tracker = CarbonTracker(epochs=1, log_dir=log_dir, logging_mode=1)

# Training loop.
for epoch in range(1):
    tracker.epoch_start()
    time.sleep(1.5)
    # Your model training.
    tracker.epoch_end()
std, out = tracker.get_logger_stream()

log = parser.parse_streams(std=std, out=out)
print(log)
log_json = json.dumps(
    log,
    indent=2,
    cls=NumpyEncoder
)
print(log['pred'])

#print(log_json)

test = parser.filter_logs(log['components'], ['epoch_durations (s)'], nested=True)
print(test)
# Optional: Add a stop in case of early termination before all monitor_epochs has
# been monitored to ensure that actual consumption is reported.
tracker.stop()


