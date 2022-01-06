# Background Edit with GLIDE + DeepLabv3

<!-- ABOUT THE PROJECT -->
## About The Project

This project involves the editing of the background of any image while preserving the foreground scene, object, or people. It combines the ability of the GLIDE model to perform image inpainting and an off-the-shelf image segmentation model like DeepLabv3.

<br/>

## Results

| Original    | On the beach      | Star Wars      | Effiel Tower     |
|------------|-------------|------------|-------------|
| <img src="./images/original.jpg" width="100" height="100"/>  |<img src="./images/beach woman.png" width="100" height="100"/> | <img src="./images/star wars.png" width="100" height="100"/> | <img src="./images/effiel tower.png" width="100" height="100"/> |


<br/>


## Getting Started

You can run the [demo](demo.ipynb) on Google Colab.

or 

To get a local copy up and running, follow these simple example steps.
<br/><br/>

### Prerequisites

* PyTorch (version 1.10.0)

  Install using Anaconda:
  ```sh
  conda install pytorch=1.10.0 -c pytorch
  ```
* CUDA (version 11)

<br/>

### Installation

1. Install the official prebuilt GLIDE package.
   ```sh
   pip install -q git+https://github.com/openai/glide-text2im
   ```

2. Clone the official GLIDE repo
   ```sh
   git clone https://github.com/openai/glide-text2im.git
   ```
<br/>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
<br></br>


<!-- CONTACT -->
## Contact

Stephen Lau - [Email](stephenlaulh@gmail.com) - [Twitter](https://twitter.com/StephenLLH) - [Kaggle](https://www.kaggle.com/faraksuli)


