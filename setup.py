import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name         = 'fastautoml',  
     version      = '0.7',
     scripts      = ['fastautoml/fastautoml.py'] ,
     author       = "R. Leiva",
     author_email = "rgarcialeiva@gmail.com",
     description  = "Fast Auto Machine Learning",
     long_description = long_description,
     long_description_content_type="text/markdown",
     url         = "https://github.com/rleiva/fastautoml",
     packages    = setuptools.find_packages(),
     classifiers = [
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
         "Operating System :: OS Independent",
     ],
 )
