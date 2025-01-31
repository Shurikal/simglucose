from setuptools import setup

setup(
    name='simglucose',
    version='0.3.0',
    description=
    'A Type-1 Diabetes Simulator as a Reinforcement Learning Environment in OpenAI gym or rllab (python implementation of UVa/Padova Simulator)',
    url='https://github.com/jxx123/simglucose',
    author='Jinyu Xie',
    author_email='xjygr08@gmail.com',
    license='MIT',
    packages=['simglucose'],
    install_requires=[
        'pandas', 'numpy', 'scipy', 'matplotlib', 'pathos', 'gymnasium'
    ],
    include_package_data=True,
    zip_safe=False,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
