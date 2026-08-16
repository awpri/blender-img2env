"""Photo3D solve daemon.

Runs outside Blender in its own venv. Blender's bundled Python must never get
torch installed into it — that is a recurring source of broken installs, and it
would reload 1.9 GB of weights every time Blender restarts.

Submodules are imported lazily by solver_server.py so that `import server.exif`
in a test does not drag in torch.
"""

__version__ = "0.8.0"
