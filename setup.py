from setuptools import setup

setup(
    name="minkasi_jax",
    version="1.2.0",
    install_requires=["numpy", "astropy", "jax", "jaxlib"],
    extras_require={
        "fitter": [
            "pyyaml",
            "minkasi",
        ]
    },
)
