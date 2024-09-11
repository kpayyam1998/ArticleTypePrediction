from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT="-e ."
def get_requirements(file_path:str)->List[str]:
    requirement=[]
    with open(file_path) as obj:
        requirement=obj.readlines()
        requirement=[req.replace("\n","") for req in requirement]

        if HYPEN_E_DOT in requirement:
            requirement.remove(HYPEN_E_DOT)
    return requirement

setup(
    name="ArticlesTypePrediction",
    version="0.0.0",
    author="kp",
    author_email="kp@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")

)