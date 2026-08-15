"""Fail fast unless Isaac Sim can create and render one headless frame."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
try:
    print("ISAAC_APP_READY", flush=True)
    app.update()
    print("ISAAC_RENDER_FRAME_OK", flush=True)
finally:
    app.close()
