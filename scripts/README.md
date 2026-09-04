# Scripts

| Script | Purpose |
|---|---|
| `generate_model.py` | build the MuJoCo model from the four spiral parameters |
| `render_demo.py` | headless video/GIF of the model curling (EGL) |

```bash
uv run scripts/generate_model.py
uv run scripts/render_demo.py --seconds 8 --gif
uv run scripts/render_demo.py --controller replay --seconds 30   # measured forces
```

`render_demo.py` needs the `vision` extra (`uv pip install -e ".[vision]"`) for
the video encoder. It frames the camera on the swept volume: it runs the excitation
once to collect the bounding box of every body position, then centres on that.
So the framing adapts when you change the model or the excitation.

Output goes to `build/media/`.
