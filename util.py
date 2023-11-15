
def log_cube(cubePlacement, success, viz, name="log_box"):
    viz.addBox(name, [0.1, 0.1, 0.1], [0.0, 1.0, 0.0] if success else [1.0, 0.0, 0.0])
    viz.applyConfiguration(name, cubePlacement)
