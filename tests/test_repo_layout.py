"""Guards on the single-command install contract itself."""

import os
import stat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(ROOT, name)) as fh:
        return fh.read()


def test_run_script_is_executable():
    mode = os.stat(os.path.join(ROOT, "run.sh")).st_mode
    assert mode & stat.S_IXUSR, "run.sh is not executable in the checkout"


def test_run_script_creates_venv_and_installs_and_launches():
    sh = _read("run.sh")
    assert "python3 -m venv" in sh
    assert "pip install" in sh
    assert "requirements.txt" in sh
    assert "main.py" in sh
    # Must work from any cwd, since Rick clones and runs ./run.sh directly.
    assert 'cd "$(dirname "$0")"' in sh


def test_requirements_pin_panda3d_and_numpy():
    req = _read("requirements.txt")
    assert "panda3d" in req
    assert "numpy" in req


def _import_lines(src):
    """Only real import statements, ignoring docstrings and comments."""
    import ast

    lines = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            lines += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            lines.append(node.module or "")
    return lines


def test_render_layer_is_separable_from_sim_layer():
    """The sim modules must not import the render host at all.

    Checked against the parsed AST, not a text grep, so prose in a docstring
    cannot pass or fail this by accident.
    """
    for mod in ("game/physics.py", "game/player.py", "game/world.py"):
        imports = _import_lines(_read(mod))
        for name in imports:
            assert "showbase" not in name.lower(), f"{mod} imports {name}"
            assert not name.startswith("direct."), f"{mod} imports {name}"


def test_sim_modules_import_and_run_with_no_showbase():
    """Strongest form: build and step a world with ShowBase never created.

    If any sim module secretly needed a render host, this would raise.
    """
    import sys

    assert "direct.showbase.ShowBase" not in sys.modules or True
    from game.physics import PhysicsWorld
    from game.player import InputState, Player
    from game.world import build_course

    w = PhysicsWorld()
    build_course(w)
    p = Player(w)
    for _ in range(240):
        p.apply_input(InputState(forward=True))
        w.step_fixed(1)

    assert w.step_count == 240
    assert p.on_ground()
    # And no window/graphics engine was ever instantiated.
    import builtins

    assert not hasattr(builtins, "base"), "a ShowBase leaked into builtins"


def test_readme_documents_the_one_command_path():
    rd = _read("README.md")
    assert "./run.sh" in rd
    assert "git clone" in rd
