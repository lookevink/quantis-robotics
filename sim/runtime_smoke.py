"""Fail fast unless Isaac Sim can create and render one headless frame."""

import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
print("ISAAC_APP_READY", flush=True)
app.update()
print("ISAAC_RENDER_FRAME_OK", flush=True)

# Isaac Sim 6 can still have asynchronous UI tasks during the first frame, and
# its close path asserts if those tasks race extension teardown. This process
# has no state to flush, so exit after proving startup and one rendered frame.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
