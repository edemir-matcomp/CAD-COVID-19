<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->



<!-- PROJECT LOGO -->

# Auxiliando o combate do COVID-19

<!--
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.svg" alt="Logo">
  </a>
</p>

O projeto CADCOVID-19 tem a proposta de oferecer um sistema online para auxiliar no diagnóstico da COVID-19 utilizando técnicas de inteligência artificial e visão computacional.
-->



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage-instructions">Usage Instructions</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#authors">Authors</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

The proposal of the CADCOVID-19 project is to provide health professionals with a system capable of assisting the diagnosis of pulmonary diseases from images, focusing on cases related to the outbreak of COVID-19. Its main objective is to offer an online system for the centralization of x-ray and tomography data of patients diagnosed with COVID-19 or suspected cases.

With the aid of computer vision and artificial intelligence algorithms, the system will allow researchers and health professionals to upload medical images, which will be integrated into the repository. From these images a report will be generated with proprieties extracted from them, in order to assist the diagnosis of lung diseases.

In this specific repository, you will find the source codes for the following tasks: 

* **Auxílio a Diagnóstico de Lesões Pulmonares** - [1_Diagnostic_Aid](https://github.com/edemir-matcomp/CAD-COVID-19/tree/master/1_Diagnostic_Aid)
* **Segmentação Semântica de Pulmões** - [2_Lung_Segmentation](https://github.com/edemir-matcomp/OLACEFS_DAM/tree/master/2_Lung_Segmentation)
* **Detecção de Lesões Pulmonares** - [3_Lesion_Detection](https://github.com/edemir-matcomp/OLACEFS_DAM/tree/master/3_Lesion_Detection)

### Built With

This section shows a list of the major frameworks that was used to built this project

* [Pytorch](https://pytorch.org/)
* [Flask](https://flask.palletsprojects.com/en/2.0.x/#)


<!-- GETTING STARTED -->
## Getting Started

These are instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

What things you need to install the software and how to install them

```
Docker Engine: https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository
Nvidia-Docker (for gpu support)
```

### Installation

```
# Docker installing steps
sudo apt-get install     apt-transport-https     ca-certificates     curl     software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo docker run hello-world

# Nvidia-Docker installing steps
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |   sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo reboot
```

## Running the docker container

```
# With GPU support
docker run -it --gpus all --ipc=host --name=cadcovid_container -w /home -v /home:/home edemirfaj/patreo_bot:gpu-py3 bash

# Without GPU support
docker run -it --ipc=host --name=cadcovid_container -w /home -v /home:/home edemirfaj/patreo_bot:gpu-py3 bash
```

## Usage Instructions

Detailed instructions for each of the project deliveries are in their respective folders
              
1. **Auxílio a Diagnóstico de Lesões Pulmonares** - [1_Diagnostic_Aid](https://github.com/edemir-matcomp/CAD-COVID-19/tree/master/1_Diagnostic_Aid), [COLAB](https://colab.research.google.com)
2. **Segmentação Semântica de Pulmões** - [2_Lung_Segmentation](https://github.com/edemir-matcomp/OLACEFS_DAM/tree/master/2_Lung_Segmentation), [COLAB](https://colab.research.google.com)
3. **Detecção de Lesões Pulmonares** - [3_Lesion_Detection](https://github.com/edemir-matcomp/OLACEFS_DAM/tree/master/3_Lesion_Detection), [COLAB](https://colab.research.google.com)

<!-- DISCLAIMER -->
## Disclaimer

This api-rest and accompanying pretrained models are provided with no guarantees regarding their reliability, accuracy or suitability for any particular application and should be used for research purposes only. The models and code are not to be used for public health decisions or responses, or for any clinical application or as a substitute for medical advice or guidance.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Jefersson Alex - [Website](https://homepages.dcc.ufmg.br/~jefersson/) - jeferssonalex@gmail.com 

Project Website: [http://www.cadcovid19.dcc.ufmg.br/](http://www.cadcovid19.dcc.ufmg.br/)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgments

* Fundação de Amparo a Pesquisa do Estado de Minas Gerais - FAPEMIG - APQ-00519-20

## Authors
* **Alexei Machado** - [alexeimachado](https://github.com/alexeimachado)
* **Camila Laranjeira** - [camilalaranjeira](https://github.com/camilalaranjeira)
* **Cristiano Rodrigues** - [cristianorbh](https://github.com/cristianorbh)
* **Diego Campos** - [diegohaji](https://github.com/diegohaji)
* **Edemir Ferreira** - [edemir-matcomp](https://github.com/edemir-matcomp)
* **Ester Fiorillo** - [esterfiorillo](https://github.com/esterfiorillo)
* **Hugo Oliveira** - [hugo-oliveira](https://github.com/hugo-oliveira)
* **Marcos Vendramini** - [marcosvendramini](https://github.com/marcosvendramini)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/edemir-matcomp/CAD-COVID-19/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/edemir-matcomp/CAD-COVID-19/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/edemir-matcomp/CAD-COVID-19/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/edemir-matcomp/CAD-COVID-19/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/edemir-matcomp/CAD-COVID-19/blob/master/LICENSE.txt
<!--
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
-->
[product-screenshot]: images/website_front.resized.png
