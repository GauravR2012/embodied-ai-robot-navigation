from setuptools import setup

package_name = "navigation_controller"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "go_to_goal = navigation_controller.go_to_goal:main",
        ],
    },
)
