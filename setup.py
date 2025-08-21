from setuptools import find_packages, setup
from typing import List
HYPEN_E_DOT ='-e .'

def get_requirements(file_path:str)->List[str]:
    """This Function returns a list of packages 
    from requiremtnts.txt to installfor this project
    """
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n'," ") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='Ham_Spam_Classification_ML_Project',
    version='0.1',
    author='Rohit',
    author_email='stat.data247@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)