# File: src/toolkit/modules/carbon_tracker.py

import os
from codecarbon import EmissionsTracker

class CarbonTracker:
    """Wrapper around CodeCarbon’s EmissionsTracker."""

    def __init__(
        self,
        project_name: str = "default_project",
        output_dir: str = "carbon_logs",
    ):
        # ensure output folder exists
        os.makedirs(output_dir, exist_ok=True)
        # now project_name and output_dir are defined
        self.tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=output_dir,
            log_level="error",
        )

    def start(self):
        self.tracker.start()

    def stop(self):
        self.tracker.stop()
