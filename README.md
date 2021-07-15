<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.svg" alt="Logo">
  </a>
  <!--
  <h3 align="center">CAD-COVID 19</h3>
  -->
  <p align="center">
    Auxiliando o combate do COVID-19
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="http://www.cadcovid19.dcc.ufmg.br/">View Project Website</a>
  </p>
</p>



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
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

There are many great README templates available on GitHub, however, I didn't find one that really suit my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need -- I think this is it.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should element DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people have have contributed to expanding this template!

A list of commonly used resources that I find helpful are listed in the acknowledgements.

### Built With

This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Bootstrap](https://getbootstrap.com)
* [JQuery](https://jquery.com)
* [Laravel](https://laravel.com)



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

What things you need to install the software and how to install them

```
Docker Engine: https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository
Nvidia-Docker (for gpu support)
```

### Installing

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
docker run -it --gpus all --ipc=host --name=olacef_container -w /home -v /home:/home edemirfaj/patreo_bot:gpu-py3 bash

# Without GPU support
docker run -it --ipc=host --name=olacef_container -w /home -v /home:/home edemirfaj/patreo_bot:gpu-py3 bash
```

### Instructions

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

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Website: [http://www.cadcovid19.dcc.ufmg.br/](http://www.cadcovid19.dcc.ufmg.br/)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgments

* Organização Latino-Americana e do Caribe de Entidades Fiscalizadoras Superiores (OLACEFS)

## Authors

* **Cristiano Rodrigues** - [edemir-matcomp](https://github.com/edemir-matcomp)
* **Diego Campos** - [edemir-matcomp](https://github.com/edemir-matcomp)
* **Edemir Ferreira** - [edemir-matcomp](https://github.com/edemir-matcomp)
* **Ester Fiorillo** - [esterfiorillo](https://github.com/esterfiorillo)
* **Hugo Oliveira** - [edemir-matcomp](https://github.com/edemir-matcomp)
* **Marcos** - [edemir-matcomp](https://github.com/edemir-matcomp)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/website_front.resized.png
